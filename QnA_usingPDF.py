import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import numpy as np
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer
import torch
import re


#api_key='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'
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key=api_key


global faiss_vectorstore

class ScalableSemanticSearch:
    """Vector similarity using product quantization with sentence transformers embeddings and cosine similarity."""

    def __init__(self, device="cpu",model="sentence-transformers/all-mpnet-base-v2"):
        self.device = device
        self.model = SentenceTransformer(
            model, device=self.device
        )
        self.model_name=model
        #"distilbert-base-nli-stsb-mean-tokens"
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.quantizer = None
        self.index = None
        self.hashmap_index_sentence = None

  

    @staticmethod
    def load_data(file):
      pdf=pdfplumber.open(file)
      pdf_text=[]
      for page in pdf.pages:
        page_text=page.extract_text()
        pdf_text.append(page_text)
      doc_text='\n'.join(pdf_text)
      return doc_text


    def build_vectorstore(self, doc_text):
        """Build the index for FAISS search.

        Args:
            embeddings: Numpy array of encoded sentences.
        """
        
        model_name = self.model_name
       
        if torch.cuda.is_available():
          model_kwargs = {"device": "cuda"}
        else:
          model_kwargs = {"device": "cpu"}

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_splits = text_splitter.split_text(doc_text) 
        
        embeddings = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs)
        
        vectorstore = FAISS.from_texts(all_splits, embeddings)
        
        return vectorstore


search_model=ScalableSemanticSearch(model="bert-base-nli-mean-tokens")
doc_text=search_model.load_data('')
faiss_vectorstore=search_model.build_vectorstore(doc_text)

class QnA(ScalableSemanticSearch):

  faiss_vectorstore= faiss_vectorstore

  def __init__(self,question):
    self.question=question
    
  def find_context(self):

    match_text=self.faiss_vectorstore.similarity_search(self.question)
    new_line_garbage_patt=re.compile(r'\\n')
    sentences=[]
    for matches in match_text:
  
      raw_string=str(matches)[14:-14]
      #print(raw_string)
      sentence=re.sub(new_line_garbage_patt,'\n',raw_string)
      sentences.append(sentence)
      
    complete_matches=' '.join(sentences)
    return complete_matches

  def respond(self,doc_text):


    try:
      prompt=f"Question:{self.question}\nContext:{doc_text}\nAnswer:"
      response=openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      temperature=0.1,
      max_tokens=512)

      
      answer=response['choices'][0]['text'].strip()
      return answer
    except openai.error.RateLimitError:
      return -1
    
query=""
qna_obj=QnA(query)
doc_text=qna_obj.find_context()
answer=qna_obj.respond(doc_text)


