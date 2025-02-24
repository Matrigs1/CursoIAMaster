import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


os.environ['OPENAI_API_KEY'] = 'SUA CHAVE DE API'

model = ChatOpenAI(model='gpt-3.5-turbo')

template = '''
Traduza o texto do {idioma1} para o {idioma2}:
{texto}
'''

prompt_template = PromptTemplate.from_template(
    template=template
)

prompt = prompt_template.format(
    idioma1='português',
    idioma2='francês',
    texto='Bom dia!',
)

response = model.invoke(prompt)

print(response.content)
