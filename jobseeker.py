from selenium import webdriver
from datetime import datetime, timedelta
import json
import time
from selenium.common import NoSuchElementException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import os
import torch
torch.cuda.empty_cache()
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

### Generate
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

from langchain_community.embeddings import GPT4AllEmbeddings


class JobSeeker():
    def __init__(self,jobsite):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        self.jobsite = jobsite
        self.driver = webdriver.Chrome(options=options)
        self.driver.get("https://www.google.com")
        try:
            self.driver.find_element(By.XPATH, '//div[text()="Accept all"]').click()
        except Exception as e:
            print("Could not find or click the accept button:", e)

    #machine learning engineer "london" after: 2024 - 05 - 01 site: lever.co
    def get_jobs(self, job_title:str, location:str=None, date:str=None):
        links = set()
        wait = WebDriverWait(self.driver, 10)
        if date == None or not self._is_valid_date(date):
            today = datetime.today()
            yesterday = today - timedelta(days=1)
            date = yesterday.strftime('%Y-%m-%d')

        self.driver.get("https://www.google.com")
        search_box = self.driver.find_element(By.NAME, 'q')
        location_str = ''
        if location:
            location_str = f'"{location}"'
        search_query = f'{job_title} {location_str} after:{date} site:{self.jobsite}'
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)
        all_sites = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'h3')))
        for site in all_sites:
            link = site.find_element(By.XPATH, './ancestor::a').get_attribute('href')
            if link is not None and f'{self.jobsite}/' in link:
                if link.endswith("/apply"):
                    link = link[:-6]
                links.add(link)

        return list(links)


    def _is_valid_date(self,date_string, date_format='%Y-%m-%d'):
        try:
            # Attempt to parse the date string
            datetime.strptime(date_string, date_format)
            return True
        except ValueError:
            # If parsing fails, the string is not a valid date
            return False

    def close_driver(self):
        self.driver.close()

class Lever(JobSeeker):
    def __init__(self):
        super(Lever,self).__init__('lever.co')

class LeverApplier():
    def __init__(self,link):
        #super(LeverApplier,self).__init__('lever.co')
        options = webdriver.ChromeOptions()
        #options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)
        self.link = link
        self.fetch_job_details()
        self._initialise_rag()

    def fetch_job_details(self):
        self.driver.get(self.link)
        wait = WebDriverWait(self.driver, 10)
        self.title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'h2'))).text
        self.job_desc = wait.until(EC.presence_of_element_located((By.XPATH, "//div[@data-qa='job-description']/ancestor::div")))

    def fill_application(self):
        self.driver.get(self.link+"/apply")
        wait = WebDriverWait(self.driver, 10)
        #We first attach the document, this will fill trivial values in for us:
        try:
            file_path = os.path.abspath("./resume_CV.pdf")
            wait.until(EC.presence_of_element_located((By.ID, 'resume-upload-input'))).send_keys(file_path)
        except ValueError:
            print("Couldn't upload resume")

        time.sleep(8)
        all_questions = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'application-question')))

        for question in all_questions:
            if self._check_resume(question):
                continue
            select = self._check_select(question)

            try:
                label = question.find_element(By.CLASS_NAME, 'application-label')
                options = question.find_element(By.CLASS_NAME, 'application-field')
                if self._check_filled(options):
                    continue
                if self._check_long_question(question):
                    response = self._answer_question_long_question(label.text)
                    question.find_element(By.XPATH, './/textarea').send_keys(response)
                else:
                    response = self._answer_question(label.text, options.text,select)
                    self.complete_field(question, response)
            except NoSuchElementException:
                pass
        #btn-submit
        self.driver.find_element(By.ID, 'btn-submit').click()

    def complete_field(self,question, response):
        locators = [
            (By.XPATH, './/input[@type="checkbox"]'),
            (By.XPATH, './/input[@type="radio"]'),
            (By.TAG_NAME, 'select'),
            (By.XPATH, './/input[@type="text"]')
        ]

        for locator in locators:
            try:
                element = question.find_elements(*locator)
                for e in element:
                    if locator[1] == './/input[@type="checkbox"]':
                        if e.get_attribute("value") in response.get('response', ''):
                            e.send_keys(Keys.SPACE)
                    elif locator[1] == './/input[@type="radio"]':
                        if e.get_attribute("value") in response.get('response', ''):
                            e.send_keys(Keys.SPACE)
                    elif locator[1] == 'select':
                        select = Select(e)
                        select.select_by_visible_text(response.get('response', ''))
                    elif locator[1] == './/input[@type="text"]':
                        e.send_keys(response.get("response", ""))
                if len(element)>0:
                    break
            except NoSuchElementException:
                pass
    def _check_filled(self,options):
        try:
            value = options.find_element(By.XPATH, './/input[@type="text" or @type="email"]').get_attribute("value")
            print(value)
            if value != "":
                return True
            else:
                return False
        except NoSuchElementException:
                return False

    def _check_long_question(self,question):
        try:
            if question.find_element(By.XPATH, './/textarea') != None:
                return True
            else:
                return False
        except NoSuchElementException:
                return False

    def _check_resume(self, question):
        try:
            if question.find_element(By.ID, 'resume-upload-input') != None:
                return True
            else:
                return False
        except NoSuchElementException:
            return False

    def _check_select(self, question):
        try:
            if question.find_element(By.TAG_NAME, 'select') != None:
                return True
            else:
                return False
        except NoSuchElementException:
            return False

    def _answer_question(self,asked_question,options,select=False):
        llama_llm = HuggingFacePipeline(pipeline=self.pipeline)

        # Create the Prompt template
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a spanish person applying for a job.
            You will be asked a question regarding your personal information. Based in the context provided and your
            willingness to get an interview respond the questions effectively. You will be given some context and personal information.
            If options are given just return only one of the given options. 
            Provide the option as a JSON with a single key 'response' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Context: {context} 
            Personal Information: {personal_info}
            Question: {question} 
            Options: {options}
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["context", "personal_info","question","options"],
        )
        #Read personal info
        with open('info.json', 'r') as file:
            data = json.load(file)

        if select:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=50, chunk_overlap=10
            )
            splits = text_splitter.create_documents(options.splitlines())
            vectorstore = Chroma.from_documents(
                documents=splits,
                collection_name="select-collection",
                embedding=GPT4AllEmbeddings(),
            )
            opt_retriever = vectorstore.as_retriever()
            options = opt_retriever.invoke(asked_question+str(data))

        # Create llm chain
        docs = self.retriever.invoke(asked_question)
        # Chain
        rag_chain = prompt | llama_llm | JsonOutputParser()
        # docs.append(web_search_response["context"])
        generation = rag_chain.invoke({"context": docs,"personal_info":data, "question": asked_question,"options":options})
        return generation


    def _answer_question_long_question(self,asked_question):
        llama_llm = HuggingFacePipeline(pipeline=self.pipeline)

        # Create the Prompt template
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a spanish person applying for a job.
            You will be asked a question regarding your personal information. Based in the context provided and your
            willingness to get an interview respond the questions effectively. 
            Your objective is to look at the personal information (Resume) and some examples of previous cover letters, try to answer
            the given question with a similar style to the given examples. 
            Base your response in the specific job description and the background information provided avoid any preamble.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Resume: {context} 
            Examples: {examples}
            Job description: {job_desc}
            Question: {question}
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["context", "examples", "job_desc","question"],
        )
        #Read personal info
        with open('info.json', 'r') as file:
            data = file.read()

        # Create llm chain
        docs = self.retriever.invoke(asked_question)
        # Chain
        rag_chain = prompt | llama_llm | StrOutputParser()

        generation = rag_chain.invoke({"context": docs,"examples":data,"job_desc":self.job_desc, "question": asked_question})
        return generation

    def _initialise_rag(self):

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        quantization_config = BitsAndBytesConfig(
            #load_in_4bit=True,
            load_in_4bit=True,
            #bnb_4bit_quant_type="nf8",
            #bnb_4bit_compute_dtype="bfloat16",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  )
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     torch_dtype=torch.bfloat16,
                                                     quantization_config=quantization_config,
                                                     )
        self.pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.01,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=300,
        )

        file_path = os.path.abspath("./CV.pdf")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits,
            collection_name="rag-chroma",
            embedding=GPT4AllEmbeddings(),
        )
        self.retriever = vectorstore.as_retriever()


