from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0)

txt_file = 'electric_vehicle.txt'

with open(txt_file, 'r') as file:
    txt = file.read()

print(llm.get_num_tokens(txt))

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=8000, chunk_overlap=200)

docs = text_splitter.create_documents([txt])

print(len(docs))

summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)

summary = summary_chain.run(docs)
print(summary)