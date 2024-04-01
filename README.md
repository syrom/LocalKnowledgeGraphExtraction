# Motivation and context
As usual, the way is the destination. So besides just providing the code part, I enter here into the details of the WHY I have written this code. If you are just interested in the result and the code, feel free to directly plunge into the code. But you’ll miss important context information …. which makes a perfect transition to the graph subject:

## Why Graphs ?
I already worked with graphs before, in the context of a more flexible visualization of organisational realities beyond pure hierarchy.

This interest was renewed after toying around with RAG (retrieval augmented generation). Graphs as basis for RAG become interesting in use cases where purely statistical word embeddings are not good enough and are conducive to have the LLM produce halucination.

Graphs provide safer ground for RAG by providing better context because they encapsulate the semantics contained in text and are a good way to foster understanding thru (a) potential visualization of semantic relationships and (b) re-use as input for graph embeddings (vs. pure word embeddings).

However, full RAG is a second step. This article is limited to creating the graph from unstructured text. Embedding graphs and using them for semantic-driven RAG is out of scope and reserved for a future article.

## Why a local LLM ?
I had never run an LLM locally — but clearly saw the interest and need:
- Privacy and confidentiality, e.g. when treating company information that cannot be transmitted to uncontrolled third parties. The example in this code is one that would not require this level of confidentiality. But it serves as a proof of concept that shows a solution that is also fit e.g. to extract knowlege from confidential company documents, as this info does not leave your computer.
- Use and support open source models to foster informational autarchy and independence from “winner-takes-it-all” companies. If you are interested why this is important, I recommend to read this article from Daniel Jeffries (and eventually subscribe to his substack channel).
- Converging hardware and software evolution, making it possible to run high quality LLMs on (comparably) affordable home solution.

The concrete choice fell on combining Ollama, Mixtral 8x7B Mixture-of-Expert model on Macbook M1 Pro. Ollama promised to provide an easy way to make LLMs run locally in a very easy way. And I was not disappointed — it’s truly amazingly simple.

I heard that Mixtral shows VERY good performance, at par e.g. with chatGPT 3.5. Calculation and feedback from the internet suggested the model must be able to run on a Mac with 64 GB RAM. And the price drop of used M1 hardware after the introduction of M3 processors made this choice available to me by only “sacrificing” 25% — 30% of performance vs. the new M3-benchmark at less than 50% of the price — and performance (as I found out later) still largely sufficient for own use cases, even though the tasks are VERY calculation itensive.

Incidently, the calculation intensity of the task would also drive up the costs of any 3rd-part API use to much higher levels — so even when I haven’t done the math yet, I assume that the investment in own hardware is also cost efficient in the end. This, of course, under the assumption that you actually use it in the long run. But graph extraction is not the only use case. I also see an ample field for local agents to provide support on daily tasks. How this can look in concrete terms, I already tried out with the “career coach agent” — just that the code, at the time, still relied on the openAI API.

## Next steps:
As mentioned above, both the knowledge extraction and the use of a local LLM lend themselves to more experiments beyond the scope of this code example.
For the use of the extracted graph, this means primarily the use as basis for improved, semantic based RAG.Additional uses for running an LLM locally are (1) the possibility to finetuning the open-source models with own data, helping the LLM to provide better answers that are more relevant to address the given use case for which data is available and (2) the mentioned possiblity to run agent frameworks locally.Details on the use case: History of the Germans podcast transcripts

## Use case treated in the code example
As a first use case, I targeted to extract a knowledge graph from unstructured text from my currently favourite podcast “The History of the Germans” by Dirk Hoffmann-Becking: a sincere and true recommendation for any History buff.
History of the Germany Podcast on Spotify History of the Germany Podcast on Spotify
I already scraped the the podcast transcript from the excellent associated website, with one large text corpus per period (e.g. “Ottonians”, “Salians”, “Hohenstaufen” etc.). However, for reasons explained below, this example only works on the transcript of a single episode.
History texts show very well the shortcomings of “embedding only” RAG, motivating the interest in a more semantic-driven knowledge-graph based RAG to query the text (see: “Next steps” above).
The proof for this: I already created a GPTs based on the transcript corpus. But querying the text afterwards showed very mixed results.
Chronology and relationships are obviously very important concept in a History text — but word embdeddings do not capture well these concepts.
A good example: as strange as it may sound nowadays, but in the periods covered in the corpus, excommunication of emperors thru popes was a powerful political tool that was put to use on a regular basis (…and any self-respecting emperor wouldn’t suffer NOT being excommunicated at least once…). But it matters, of course, if Pope P1 excommunicates Emperor E1 or Emperor E2. Especially if Emperor E2 happens to be the great-granson of Emperor E1 and Pope P1 was alread pushing up daisies since decades before the rule of Emperor E2 started.

Word embeddings do capture well the relationship “Popes excommunicate Emperors”…. but they start too quickly halcuinating the corresponding names (e.g.: if Pope P1 excommunicated Emperor E1, why shouldn’t he have excommunicated Emperor E2 ?). Precisely because the embeddings cannot expressly capture the chronological or relational aspects linking the words they embed.

Establishing this link literally means establishing a graph. In a knowledge graph representation, there would only be “edges” from Pope P1 to Emperor E1, not to E2 (because they are seperated by a lifetime, preventing any co-occurences).

Voilà, my example why I want to test out a knowledge-graph based RAG

And as a first step, this means being able to establish this graph (and visualise it, as visualisation is a very good way to understanding)

In the concrete example here on GitHub, the code uses the transcript of Episode 96 “Saxony and Eastwards expansion: Meet the neighbours”.

## The Hackathon
So there was my intent… what was missing was the opportunity. Which came around in the form of a Python Hackathon of the Düsseldorf Python User Group PyDDF, organized, chaired and driven by Marc-André Lemburg and Charly Clark (amongst others things maintainer of the openpyxl package). If you are interested in more info on this group, consult their webpage or their YouTube-channel.

Prior to the Hackathon weekend, I did some research and found, incidentally, an excellent article here on Medium that had the potential to take me 80%-90% of the way to my objective.

So the task for the hackathon was to understand and modify the code from this article to be able to extract a knowledge graph, encapsulating semantic information, from parts of the “History of the Germans” podcast transcript to serve as future input for a graph-based RAG chat with this content.

## The article that inspired it all — and changes to it
As said, the inspirational article I found provides an excellent basis, showing how to achieve precisely the initial intent: extract and visualize a knowledge graph from unstructured text:
“How to convert any text into a graph of concepts” by Rahul Nayak on medium.com.
The changes and modifications done to the code from this example were mainly:
- Transformation to a single All-in-One notebook
Use of different LLM (using now the stronger Mixtral model)
- Elimination of some seemingly unused function defintions from the code base
- Making relevant adjustments to the SYS_PROMPT due to the History use case
The last point taught me a lot about prompting: the SYS_PROMPT is the real prompt, whereas USER_PROMPT is actually less of a prompt, but rather (comparable to RAG) context-info that the SYS_PROMPT performs tasks on.

And this SYS_PROMPT needed careful revision according to the altered use case: The inspirational article focused on articles on the health system in India. That is a domain quite different from medival German history. A first run yielded disappointing result… until I checked every line in the instructions contained in the SYS_PROMPT: e.g. the identification of persons as entities was expressly exluded from the concept extraction prompt. Which produces quite some limitiations for texts covering History. Results improved a lot after adjusting the SYS_PROMPT to the covered field of History, focusing particularly on persons as agents or entities.

The SYS_PROMPT is also a good entry point into the understanding in how far LLM based processing is different from “classical” programming: even though the instructions of the SYS_PROMPT used are clear, they do NOT produce a correct JSON output format invariably every time. One needs to check the quality of the output manually (aka the number of chunks that produce an error when trying to load the JSON-string from the LLM-prompt-call to the result list). Skipping a chunk every now and then shouldn’t be too problematic, but if the ratio of successful to unsuccesful transformations from text-chunk to JSON-format becomes too low, one should probably either work on the text input or start to modify and improve the SYS_PROMPT.

To change the LLM may come across as overkill. It would need to be tested if a smaller, more focused model wouldn’t show better efficiency. But applying the “Why do the chickens cross the road” logic (Answer: because they can !), I opted for the highest performance model that runs on the hardware described above. And that happened to be Mixtral.

## Ollama Bugs
The entire AI field is still very knew — and new software is often still in experimental stage. This also applies e.g. to Ollama. I tried to run the code above initially (over night) on the transcripts covering entire historical periods (aka dynasties like Salian, Hohenstaufen etc.) — and thus up to 40 episodes in one go. But this wouldn’t work out because Ollama would simply, at one point, stop to generate responses to the calls that the code made to Mixtral.

This bug seemed to be related to some memory overflow or leakage, because it happened after a rather constant number of generations (taking a text chunk and generating a JSON-format from it)

This bug was identified and flagged on GitHub ….. and partially fixed as of today’s Ollama update (2024–03–29):
https://github.com/ollama/ollama/issues/1863#issuecomment-2010185380
After this update, it was possible for the first time to have the code below churn thru a large text with, in the given case with > 100 chunks of a chunk-size of 1.000 characters (with 100 characters overlap).

Unfortunately, with chunk sizes > 120, I still ran inevitably into the stalling of the LLM-call: the code execution would simply stop and not return any results anymore, event though the kernel was still active. This is still good enough, though, to process the transcripts of roughly 3 podcast episodes in one batch (but, as mentioned, the GitHub example only uses the text of a single episode to make sure that it truely works).

This problem is certainly due to the novelty of all the tools used — and may or may not go away completely with further updates.

## Performance
In case you believe that local generation is done in a breeze: think again !

The performance of the knowledge extraction process is slow on a local machine (MacBook M1 Pro). Which shows how much is going on under the hood. I counted processing times of 30 sec to roughly less than a minute per chunk to produce the JSON-string with an average of around 40 sec. So a text of ca. 100 chunks (aka 100.000 character length based on a chunk size of 1000) requires over an hour of processing time to extract the knowledge graph. Plus: you better do not detach the power cord. The otherwise extremly frugal MacBook starts to consume electricity like hell once the script gets going.

Hence, the code also saves the result in several forms as CSV-files. The knowledge graph can thus later be reproduced faster, once the extraction has taken place, simply by loading the files containing the results from the extraction process. Or the output can be used as RAG-input in a 2nd step.

As said before: there is a dedicated notebook just for reproducing the knowledge graph from saved files on GitHub, skipping the time and energy intense extraction part.

