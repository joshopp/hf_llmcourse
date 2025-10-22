from transformers import pipeline



# =========== Sentiment Analysis Pipeline =======================
# classifies any sentence(s) into positive or negative sentiment
classifier = pipeline("sentiment-analysis")
clas_result = classifier(["I've been waiting for a HuggingFace course my whole life.", 
                    "I hate this so much!", 
                    "I love my girlfriend.", 
                    "<3", 
                    "I wish you were here.",
                    "You are like a cup of coffee"
                    ])
print(clas_result)


# =========== Zero Shot Classification Pipeline =======================
# classifies any sentence(s) into custom labels without any prior training on those labels (thus zero-shot)
zero_classifier = pipeline("zero-shot-classification")
zero_result = zero_classifier(
    "When going into school in the united states, you should always wear a bulletproof vest.",
    candidate_labels=["education", "political", "business","lifehack", "sarcasm", "stereotype"],
)
print("\n", zero_result)


# =========== Text Generation Pipeline =======================
# autocompletes any prompt. num_return_sequences and max_new_tokens as arguments
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
gen_result = generator(["You should quit gaming because", "you should start gaming because"], max_new_tokens=30)
gen_result = generator("In a distant future, humanity will", num_return_sequences=5, max_new_tokens=25)
print("\n", gen_result)


# =========== Mask filling Pipeline =======================
# fills the blanks (mask  tokens) in any sentence(s). Sorted by top_k -> highest probability first
unmasker = pipeline("fill-mask")
mask_result = unmasker("This course will teach you all about <mask> models.", top_k=2)
print("\n", mask_result)

unmasker_bert = pipeline("fill-mask", model="bert-base-uncased")
mask_result_bert = unmasker_bert("This course will teach you all about [MASK] models.", top_k=2)
print("\n", mask_result_bert)


# =========== Named Entity Recognition (NER) Pipeline =======================
# identifies and matches entities such as persons, organizations, locations, etc. in any sentence(s)
ner = pipeline("ner", grouped_entities=True)
ner_result = ner("My name is Joshy and I studied at the Eberhard Karls University in Tübingen.")
print("\n", ner_result)

ner_bert = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")
ner_result_bert = ner_bert("David Lama tragically died in an avalanche on 16.4.2019 while climbing in Canada.")
print("\n", ner_result_bert)


# =========== Question Answering Pipeline =======================
# answers questions based on a given context
question_answerer = pipeline("question-answering")
result = question_answerer(
    question="In which town did Joshy study?",
    context="My name is Joshy and I studied at the Eberhard Karls University in Tübingen.")
print("\n", result)


# =========== Summarization Pipeline =======================
# reduces any texts to a shorter version while preserving key information. max_length, min_length as arguments
summarizer = pipeline("summarization")
summary = summarizer(
    """
German politician Friedrich Merz says he now "regrets" comments made when asked whether he would have reservations about the appointment of a gay chancellor.
Merz is a candidate for the position of chair of the center-right Christian Democrats (CDU). Whoever is selected for the role would likely become the party's candidate to replace Angela Merkel as chancellor.
Critics said his reply, made Sunday in a video interview for the newspaper Bild, was homophobic because it made an association between being gay and pedophilia.
Speaking with the news portal t-online, Merz insisted that his statement had been "obviously misunderstood" but that if anyone was offended, "I really regret it very much."
"There was naturally malice at work," he said of the backlash, adding that colleagues had told him the criticism had been overdone.
What did Merz say?
During the original interview, Merz was asked whether he would have reservations if a gay chancellor were to lead Germany.
"No," he replied. "Concerning the question of sexual orientation, as long as it is within the law and does not affect children — which at this point, for me, would be an absolute limit — it is not an issue for public discussion."
Merz' answer sparked an outcry from German politicians, especially as one of his rivals for the position of CDU chair, Health Minister Jens Spahn, is openly gay.
Spahn told a reporter: "If the first things you associate with homosexuality are questions of law and pedophilia, then you should rather direct your questions to Friedrich Merz."
Another gay politician, deputy chair of the center-left Social Democrats (SPD), Kevin Kühnert tweeted: "This is how someone works who cannot hide the fact that he cannot deal with the normalization of homosexuality."
Friday's backhanded apology was Merz' third attempt to qualify his comments. In an interview with Die Welt on Monday, he said the connection between homosexuality and pedophilia in response to his remarks was "maliciously constructed."
He then doubled down on his remarks, saying his tolerance always reached its limit when children are targeted and said he would "continue to say this in the future, even when clearly one or another person doesn't like it."
In a tweet later on Monday, Merz went further, saying: "I do not evaluate anyone in my work environment or among my friends and acquaintances on the basis of their sexual orientation. This is a private matter. In a liberal society, there are different ways of life."
"""
)
print("\n", summary)


# =========== Translation Pipeline =======================
# translates any sentence(s), use specified models
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")
trans_result= translator("Ich liebe es, mir an einem sonnigen Herbstnachmittag den einen oder anderen Hopfensmoothie genüsslich zu Gemüte zu führen.")
print("\n", trans_result)


# # =========== Image classification Pipeline =======================
# classifies objects in an image
image_classifier = pipeline(
    task="image-classification", model="google/vit-base-patch16-224"
)
image_result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print("\n", image_result)



# # =========== ASR Pipeline =======================
# performs automatic speech recognition (ASR) on an audio file using whisper model
transcriber = pipeline(
    task="automatic-speech-recognition", model="openai/whisper-small")
audio_result = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print("\n", audio_result)



# =========== Available Pipelines =======================
# Text pipelines
#   text-generation: Generate text from a prompt
#   text-classification: Classify text into predefined categories
#   summarization: Create a shorter version of a text while preserving key information
#   translation: Translate text from one language to another
#   zero-shot-classification: Classify text without prior training on specific labels
#   feature-extraction: Extract vector representations of text
# Image pipelines
#   image-to-text: Generate text descriptions of images
#   image-classification: Identify objects in an image
#   object-detection: Locate and identify objects in images
# Audio pipelines
#   automatic-speech-recognition: Convert speech to text
#   audio-classification: Classify audio into categories
#   text-to-speech: Convert text to spoken audio
# Multimodal pipelines
#   image-text-to-text: Respond to an image based on a text prompt