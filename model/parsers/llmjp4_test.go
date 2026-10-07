package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/harmony"
)

// LLM-jp-4 output as decoded by llama-server: every piece that follows a
// special token starts with a space, and the 4.1 template closes all but the
// last tool call of a turn with <|end|>. Inputs below omit the implicit
// "<|start|>assistant" that Init prepends.
const (
	llmjp4Reasoning  = "<|channel|> analysis<|message|> 東京の天気を調べる。<|end|>"
	llmjp4Final      = "<|start|> assistant<|channel|> final<|message|> 東京は晴れです。"
	llmjp4CallTokyo  = "<|start|> assistant to=functions.get_weather<|channel|> commentary <|constrain|>  json<|message|> {\"city\": \"Tokyo\"}"
	llmjp4CallOsaka  = "<|start|> assistant to=functions.get_weather<|channel|> commentary <|constrain|>  json<|message|> {\"city\": \"大阪\"}"
	llmjp4CallLookup = "<|start|> assistant to=functions.look_up<|channel|> commentary <|constrain|>  json<|message|> {\"q\": \"天気\"}"
	// a call as the very first message continues the implicit "<|start|>assistant" header
	llmjp4FirstCallOsaka = " to=functions.get_weather<|channel|> commentary <|constrain|>  json<|message|> {\"city\": \"大阪\"}"
)

var llmjp4Tools = []api.Tool{
	{Type: "function", Function: api.ToolFunction{Name: "get_weather"}},
	{Type: "function", Function: api.ToolFunction{Name: "look-up"}},
}

type llmjp4Result struct {
	content  string
	thinking string
	calls    []api.ToolCall
}

func runLLMJP4(t *testing.T, p Parser, chunks []string) llmjp4Result {
	t.Helper()
	p.Init(llmjp4Tools, nil, nil)
	var res llmjp4Result
	for i, chunk := range chunks {
		content, thinking, calls, err := p.Add(chunk, i == len(chunks)-1)
		if err != nil {
			t.Fatalf("Add(%q) error: %v", chunk, err)
		}
		res.content += content
		res.thinking += thinking
		res.calls = append(res.calls, calls...)
	}
	return res
}

func llmjp4Bytes(s string) []string {
	chunks := make([]string, 0, len(s))
	for i := range len(s) {
		chunks = append(chunks, s[i:i+1])
	}
	return chunks
}

func llmjp4Call(index int, name string, args api.ToolCallFunctionArguments) api.ToolCall {
	return api.ToolCall{Function: api.ToolCallFunction{Index: index, Name: name, Arguments: args}}
}

func TestLLMJP4ParserSpacedHeaders(t *testing.T) {
	cases := []struct {
		desc  string
		input string
		want  llmjp4Result
	}{
		{
			desc:  "reasoning then final",
			input: llmjp4Reasoning + llmjp4Final,
			want:  llmjp4Result{thinking: "東京の天気を調べる。", content: "東京は晴れです。"},
		},
		{
			desc:  "reasoning then single tool call, ascii arguments",
			input: llmjp4Reasoning + llmjp4CallTokyo,
			want:  llmjp4Result{thinking: "東京の天気を調べる。", calls: []api.ToolCall{llmjp4Call(0, "get_weather", args(`{"city": "Tokyo"}`))}},
		},
		{
			desc:  "tool call as the first message, japanese arguments",
			input: llmjp4FirstCallOsaka,
			want:  llmjp4Result{calls: []api.ToolCall{llmjp4Call(0, "get_weather", args(`{"city": "大阪"}`))}},
		},
		{
			desc:  "tool name mapped back to the user's spelling",
			input: llmjp4Reasoning + llmjp4CallLookup,
			want:  llmjp4Result{thinking: "東京の天気を調べる。", calls: []api.ToolCall{llmjp4Call(0, "look-up", args(`{"q": "天気"}`))}},
		},
		{
			desc:  "parallel tool calls separated by <|end|>",
			input: llmjp4Reasoning + llmjp4CallTokyo + "<|end|>" + llmjp4CallOsaka,
			want: llmjp4Result{
				thinking: "東京の天気を調べる。",
				calls: []api.ToolCall{
					llmjp4Call(0, "get_weather", args(`{"city": "Tokyo"}`)),
					llmjp4Call(1, "get_weather", args(`{"city": "大阪"}`)),
				},
			},
		},
		{
			desc:  "three parallel calls keep their order",
			input: llmjp4Reasoning + llmjp4CallTokyo + "<|end|>" + llmjp4CallLookup + "<|end|>" + llmjp4CallOsaka,
			want: llmjp4Result{
				thinking: "東京の天気を調べる。",
				calls: []api.ToolCall{
					llmjp4Call(0, "get_weather", args(`{"city": "Tokyo"}`)),
					llmjp4Call(1, "look-up", args(`{"q": "天気"}`)),
					llmjp4Call(2, "get_weather", args(`{"city": "大阪"}`)),
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc+" (whole)", func(t *testing.T) {
			got := runLLMJP4(t, &LLMJP4Parser{}, []string{tc.input})
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %+v\nwant %+v", got, tc.want)
			}
		})
		t.Run(tc.desc+" (byte by byte)", func(t *testing.T) {
			got := runLLMJP4(t, &LLMJP4Parser{}, llmjp4Bytes(tc.input))
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %+v\nwant %+v", got, tc.want)
			}
		})
	}
}

func TestLLMJP4ParserOneSpaceRule(t *testing.T) {
	cases := []struct {
		desc   string
		chunks []string
		want   string
	}{
		{desc: "one space removed", chunks: []string{"<|channel|> final<|message|> x"}, want: "x"},
		{desc: "intended leading space survives", chunks: []string{"<|channel|> final<|message|>  padded"}, want: " padded"},
		{desc: "no space, nothing removed", chunks: []string{"<|channel|> final<|message|>x"}, want: "x"},
		{desc: "space arrives in its own chunk", chunks: []string{"<|channel|> final<|message|>", " ", "x y"}, want: "x y"},
		{desc: "only the first chunk of the body is affected", chunks: []string{"<|channel|> final<|message|> a", " b"}, want: "a b"},
		{desc: "space before a partial end tag", chunks: []string{"<|channel|> final<|message|> <|en", "d|>"}, want: ""},
		{desc: "special token split across chunks", chunks: []string{"<|chan", "nel|>", " final<|mess", "age|> x"}, want: "x"},
		{desc: "newline body is kept", chunks: []string{"<|channel|> final<|message|>\nline"}, want: "\nline"},
		{desc: "lone < in the body is kept", chunks: []string{"<|channel|> final<|message|> a <", " b"}, want: "a < b"},
	}
	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			got := runLLMJP4(t, &LLMJP4Parser{}, tc.chunks)
			if got.content != tc.want {
				t.Errorf("content = %q, want %q", got.content, tc.want)
			}
		})
	}
}

func TestLLMJP4ParserParallelCallsStreamAtEachEnd(t *testing.T) {
	p := &LLMJP4Parser{}
	p.Init(llmjp4Tools, nil, nil)

	_, _, calls, err := p.Add(llmjp4Reasoning+llmjp4CallTokyo+"<|end|>", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" || calls[0].Function.Index != 0 {
		t.Fatalf("first <|end|> should close the first call, got %+v", calls)
	}

	_, _, calls, err = p.Add(llmjp4CallOsaka, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0].Function.Index != 1 || !reflect.DeepEqual(calls[0].Function.Arguments, args(`{"city": "大阪"}`)) {
		t.Fatalf("end of generation should close the last call, got %+v", calls)
	}
}

// A prefilled message continues ordinary text, not a special token, so the
// one-space rule does not apply to it.
func TestLLMJP4ParserPrefillKeepsText(t *testing.T) {
	p := &LLMJP4Parser{}
	p.Init(nil, &api.Message{Role: "assistant", Content: "東京は"}, nil)
	content, thinking, _, err := p.Add(" 晴れです。", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != " 晴れです。" || thinking != "" {
		t.Errorf("content = %q, thinking = %q", content, thinking)
	}
}

// The gpt-oss wire format has no spaces after special tokens; the parser must
// read it exactly like the harmony handler does.
func TestLLMJP4ParserReadsGPTOSSFormat(t *testing.T) {
	inputs := []string{
		"<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant<|channel|>final<|message|>Sunny.",
		"<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{\"city\": \"Tokyo\"}",
		"<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Tokyo\"}",
		"<|channel|>final<|message|>Hello\nworld",
	}
	for _, input := range inputs {
		for _, chunks := range [][]string{{input}, llmjp4Bytes(input)} {
			want := runLLMJP4(t, harmony.NewHarmonyMessageHandler(), chunks)
			got := runLLMJP4(t, &LLMJP4Parser{}, chunks)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("disagree on %q\nharmony  %+v\nllm-jp-4 %+v", input, want, got)
			}
		}
	}
}
