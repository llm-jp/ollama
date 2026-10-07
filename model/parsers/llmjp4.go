package parsers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/harmony"
)

// LLMJP4Parser handles the harmony format of LLM-jp-4: the tokenizer decodes a
// space after every special token ("<|channel|> analysis<|message|> ..."), which
// is removed before the text reaches the harmony parser, and LLM-jp-4.1 closes
// all but the last tool call of a turn with <|end|>.
type LLMJP4Parser struct {
	parser  *harmony.HarmonyParser
	nameMap *harmony.FunctionNameMap
	state   llmjp4State

	held      string // trailing text that may still turn into a special token
	dropSpace bool   // a special token was just passed on; drop one following space

	contentPrefill bool // the prompt continues a final-channel message

	toolName  string
	toolArgs  strings.Builder
	callIndex int
}

type llmjp4State int

const (
	llmjp4Content llmjp4State = iota
	llmjp4Thinking
	llmjp4ToolCalling
)

var llmjp4SpecialTokens = []string{"<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|constrain|>"}

func (p *LLMJP4Parser) HasToolSupport() bool     { return true }
func (p *LLMJP4Parser) HasThinkingSupport() bool { return true }

func (p *LLMJP4Parser) PreservedTokens() []string {
	// <|call|> ends a tool call and is left to llama-server as a stop token
	return llmjp4SpecialTokens
}

func (p *LLMJP4Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	*p = LLMJP4Parser{
		parser: &harmony.HarmonyParser{
			MessageStartTag: "<|start|>",
			MessageEndTag:   "<|end|>",
			HeaderEndTag:    "<|message|>",
		},
		nameMap: harmony.NewFunctionNameMap(),
	}

	if lastMessage != nil {
		p.parser.AddImplicitStartOrPrefill(lastMessage)
	} else {
		p.parser.AddImplicitStart()
	}
	p.contentPrefill = lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != ""

	if len(tools) == 0 {
		return tools
	}
	processed := make([]api.Tool, len(tools))
	copy(processed, tools)
	for i, tool := range processed {
		if tool.Function.Name != "" {
			processed[i].Function.Name = p.nameMap.ConvertAndAdd(tool.Function.Name)
		}
	}
	return processed
}

// Same headers as the harmony handler, with the tokenizer's space after each special token.
func (p *LLMJP4Parser) ThinkingClose() []string {
	if p.contentPrefill {
		return nil
	}
	return []string{
		"<|end|><|start|> assistant<|channel|> final<|message|>",
		"<|end|><|start|> assistant<|channel|> final <|constrain|>  json<|message|>",
		"<|end|><|start|> assistant<|channel|> commentary<|message|>",
		"<|end|><|start|> assistant<|message|>",
	}
}

func (p *LLMJP4Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	var contentSb, thinkingSb strings.Builder

	events := p.parser.AddContent(p.normalize(s, done))
	if done {
		// the last call of a turn ends with <|call|>, which stops generation before any <|end|>
		events = append(events, harmony.HarmonyEventMessageEnd{})
	}
	for _, event := range events {
		switch event := event.(type) {
		case harmony.HarmonyEventHeaderComplete:
			switch {
			case event.Header.Recipient != "":
				p.state = llmjp4ToolCalling
				p.toolName = event.Header.Recipient
			case event.Header.Channel == "analysis":
				p.state = llmjp4Thinking
			default:
				p.state = llmjp4Content
			}
		case harmony.HarmonyEventContentEmitted:
			switch p.state {
			case llmjp4Thinking:
				thinkingSb.WriteString(event.Content)
			case llmjp4ToolCalling:
				p.toolArgs.WriteString(event.Content)
			default:
				contentSb.WriteString(event.Content)
			}
		case harmony.HarmonyEventMessageEnd:
			if p.state == llmjp4ToolCalling {
				call, err := p.completeToolCall()
				if err != nil {
					return "", "", nil, err
				}
				calls = append(calls, call)
			}
			p.state = llmjp4Content
		}
	}

	return contentSb.String(), thinkingSb.String(), calls, nil
}

func (p *LLMJP4Parser) completeToolCall() (api.ToolCall, error) {
	raw := p.toolArgs.String()
	p.toolArgs.Reset()
	name := p.nameMap.OriginalFromConverted(strings.TrimPrefix(p.toolName, "functions."))
	p.toolName = ""

	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return api.ToolCall{}, fmt.Errorf("llm-jp-4 parser: tool call arguments: raw=%q: %w", raw, err)
	}
	call := api.ToolCall{Function: api.ToolCallFunction{Index: p.callIndex, Name: name, Arguments: args}}
	p.callIndex++
	return call, nil
}

// normalize removes the tokenizer's space after each special token. Text that
// could still be the beginning of a special token is held back until the next
// chunk (or the end of generation) decides.
func (p *LLMJP4Parser) normalize(s string, done bool) string {
	buf := p.held + s
	p.held = ""

	var out strings.Builder
	for buf != "" {
		idx, token := nextSpecialToken(buf)
		if idx == -1 {
			keep := 0
			if !done {
				keep = partialSpecialTokenSuffix(buf)
			}
			out.WriteString(p.stripSpace(buf[:len(buf)-keep]))
			p.held = buf[len(buf)-keep:]
			break
		}
		out.WriteString(p.stripSpace(buf[:idx]))
		out.WriteString(token)
		p.dropSpace = true
		buf = buf[idx+len(token):]
	}
	return out.String()
}

// stripSpace applies the pending one-space rule to ordinary text.
func (p *LLMJP4Parser) stripSpace(s string) string {
	if s == "" || !p.dropSpace {
		return s
	}
	p.dropSpace = false
	return strings.TrimPrefix(s, " ")
}

func nextSpecialToken(s string) (int, string) {
	idx, found := -1, ""
	for _, token := range llmjp4SpecialTokens {
		if i := strings.Index(s, token); i != -1 && (idx == -1 || i < idx) {
			idx, found = i, token
		}
	}
	return idx, found
}

// partialSpecialTokenSuffix returns how many trailing bytes of s may still
// become a special token once the next chunk arrives (e.g. "<|chan").
func partialSpecialTokenSuffix(s string) int {
	longest := 0
	for _, token := range llmjp4SpecialTokens {
		for n := min(len(token)-1, len(s)); n > longest; n-- {
			if strings.HasSuffix(s, token[:n]) {
				longest = n
				break
			}
		}
	}
	return longest
}
