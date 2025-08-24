# Use a pipeline as a high-level helper
from transformers import pipeline

#summerizer (For fisrt time run the model will be downloaded in local machine)
summerizerPipe = pipeline("summarization", model="facebook/bart-large-cnn",max_new_tokens=500)

#article
article = "Dr. A.P.J. Abdul Kalam, born in 1931 in Rameswaram, Tamil Nadu, was an Indian aerospace scientist and the 11th President of India (2002–2007) known for his pivotal role in India's space and missile programs, which earned him the titles 'Missile Man of India' and 'People/'s President'. He served as Project Director for India's first indigenous Satellite Launch Vehicle and led the development of the AGNI and PRITHVI missiles before becoming President. Kalam, a vegetarian and devout Muslim, rose from humble origins to become an inspirational figure, dedicating his life to science and education until his passing in 2015."

#response = summerizerPipe("summerize the article given as {}".format(article))

#print(response[0]['summary_text'])
import gradio as gr

def summerize(text):
    res = summerizerPipe(text)
    return res[0]['summary_text']
demo = gr.Interface(
    fn=summerize,
    inputs="text",
    outputs="text",
    title= "Text Summerizer",
    description="Provide Text To Summerize"
)
demo.launch()