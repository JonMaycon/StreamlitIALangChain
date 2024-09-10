import streamlit as st
import tempfile
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI

# Título da página
st.title('Olá, meu nome é Jon Jon 🤖')
st.subheader('Fui treinado para responder perguntas com base nas informações de documentos que me forem fornecidos!')

# Campo para inserção da chave OpenAI
openai_key = st.text_input('Insira sua chave OpenAI', type='password')

# Upload do arquivo PDF
uploaded_file = st.file_uploader('Faça upload de um arquivo PDF', type='pdf')

# Se um arquivo PDF for enviado
if uploaded_file is not None:
    # Criar um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Carregar o documento PDF usando o caminho temporário
    loader = PyPDFLoader(tmp_file_path)
    documentos = loader.load()

    # Exibir conteúdo da página de exemplo
    if len(documentos) > 5:
        st.subheader('Exemplo de conteúdo da página 5:')
        st.write(documentos[5].page_content)

    # Carregar modelo OpenAI
    if openai_key:
        chat = ChatOpenAI(model='gpt-3.5-turbo-0125', api_key=openai_key)
        chain = load_qa_chain(llm=chat, chain_type='stuff', verbose=True)

        # Campo para inserção da pergunta
        pergunta = st.text_input('Faça uma pergunta sobre o documento')

        # Botão para executar a pergunta
        if st.button('Executar Pergunta'):
            # Executar o pipeline de perguntas e respostas
            if documentos:
                result = chain.run(input_documents=documentos[:10], question=pergunta)
                st.write(result)
            else:
                st.write('Nenhum documento carregado.')
