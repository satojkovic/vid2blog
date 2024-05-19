# Chapter info
CHAPTERS_24 = """
00:00:00 intro: Tokenization, GPT-2 paper, tokenization-related issues
00:05:50 tokenization by example in a Web UI (tiktokenizer)
00:14:56 strings in Python, Unicode code points
00:18:15 Unicode byte encodings, ASCII, UTF-8, UTF-16, UTF-32
00:22:47 daydreaming: deleting tokenization
00:23:50 Byte Pair Encoding (BPE) algorithm walkthrough
00:27:02 starting the implementation
00:28:35 counting consecutive pairs, finding most common pair
00:30:36 merging the most common pair
00:34:58 training the tokenizer: adding the while loop, compression ratio
00:39:20 tokenizer/LLM diagram: it is a completely separate stage
00:42:47 decoding tokens to strings
00:48:21 encoding strings to tokens
00:57:36 regex patterns to force splits across categories
01:11:38 tiktoken library intro, differences between GPT-2/GPT-4 regex
01:14:59 GPT-2 encoder.py released by OpenAI walkthrough
01:18:26 special tokens, tiktoken handling of, GPT-2/GPT-4 differences
01:25:28 minbpe exercise time! write your own GPT-4 tokenizer
01:28:42 sentencepiece library intro, used to train Llama 2 vocabulary
01:43:27 how to set vocabulary set? revisiting gpt.py transformer
01:48:11 training new tokens, example of prompt compression
01:49:58 multimodal [image, video, audio] tokenization with vector quantization
01:51:41 revisiting and explaining the quirks of LLM tokenization
02:10:20 final recommendations
02:12:50 ??? :)
"""

# Prompt
prompt_instructions = f"""
<instructions>
You have been given images of a video at different timestamps, followed by the audio transcript in <transcript>
The transcript was generated by an AI speech recognition tool and may contain some errors/infelicities.
Your task is to transform the transcript into a markdown blog post.
This transcript is noisy. Please rewrite it using the following guidelines:
- output valid markdown
- insert section headings and other formatting where appropriate
- you are given only part of a transcript, so do not include introductory or concluding paragraphs. Only include the main topics discussed in the transcript
- use styling to make images, text, code, callouts and the page layout and margins look like a typical blog post or textbook
- remove any verbal tics
- if there are redundant pieces of information, only present it once
- keep the conversational content in the style of the transcript. Including headings to make the narrative structure easier to follow along
- the transcript includes too many images, so you should only include the most important 1-2 images in your output
- choose images that provide illustrations that are relevant to the transcript
- prefer to include images which display complete code, rather than in progress
- when relevant transcribe important pieces of code and other valuable text
- if an image would help illustrate a part of a transcript, include it
- to include an image, insert a tag with <img src="xxxxx.jpg"/> where xxxxx is replaced by the exact image timestamp inserted above the image data
- do not add any extraneous information: only include what is either mentioned in the transcript or the images

Your final output should be suitable for inclusion in a textbook.
</instructions>
"""
