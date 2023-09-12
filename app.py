import streamlit as st
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import Ollama
from langchain.schema import Document


# Título página
st.set_page_config(page_title="ChatBot IA", page_icon=":robot_face:")

# Mensagem que será apresentada por default
mensagem_abertura = "Olá, eu sou sua assistente pessoal e consigo consultar os seus documentos privados e trabalhar offline. Em que posso ajudar?"

# Função para carregar o LLM
@st.cache_data(show_spinner=False)
def carregar_llm(modelo, temperatura, tokens_contexto, top_k, top_p, repeat_penalty, num_threads):
    llm = Ollama(
        base_url="http://localhost:11434",
        model=modelo,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        num_gpu=1,
        num_thread=num_threads,
        temperature=temperatura,
        num_ctx=tokens_contexto, 
        repeat_penalty=repeat_penalty,
        top_k=top_k,
        top_p=top_p
    )

    return llm

# Função para carregar o vector store com os embeddings
@st.cache_data(show_spinner=False)
def carregar_vectorstore():
    vectorstore_path = os.path.join("./vectorstore/", indice + ".index")
    vectorstore = FAISS.load_local(vectorstore_path, HuggingFaceInstructEmbeddings(model_name="intfloat/multilingual-e5-large", cache_folder="./modelo_embedding/"))

    return vectorstore

# Função para gerar as respostas 
def obter_resposta(llm, vectorstore, pergunta, num_embeddings_retornar):
    pergunta = "The following question is in Portuguese language and you must answer in Portuguese language: " + pergunta

    conversacao = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': num_embeddings_retornar}),
        return_source_documents=True,
        memory=st.session_state.historico
    )

    resposta = conversacao({"question": pergunta})

    return resposta

# Painel lateral
with st.sidebar:
    # Nome do menu 
    st.header("Configurações LLM")
    
    # Campos do menu 
    modelo = st.selectbox("Modelo", options=["llama2","llama2:13b","llama2:70b","orca","vicuna","nous-hermes","wizard-vicuna"], index=0)
    tokens_contexto = st.slider("Tokens contexto", min_value=0, max_value=4096, value=4096, step=1)
    temperatura = st.slider("Temperatura", min_value=0.1, max_value=1.0, value=0.1, step=0.1)
    interacoes_memoria = st.number_input("Núm. interações memória", min_value=1, max_value=100, value=2, step=1)

    configuracoes_avancadas = st.checkbox("Configurações avançadas", value=False)
    top_k = 40
    top_p = 0.95
    repeat_penalty = 1.10
    num_threads = 8
    if configuracoes_avancadas:
        top_k = st.number_input("Top K", min_value=1, value=40, step=1)
        top_p = st.number_input("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
        repeat_penalty = st.number_input("Repeat Penalty", min_value=0.00, value=1.10, step=0.01, format="%.2f")
        especificar_threads = st.checkbox("Especificar threads", value=False)
        if especificar_threads:
            num_threads = st.number_input("Núm. Threads", min_value=1, value=1, step=1)

    # Espaço entre os menus
    st.markdown("#")
    
    # Nome do menu 
    st.header("Selecionar Embedding")  

    # Campos do menu 
    indice_options = sorted([dir for dir in os.listdir("./vectorstore") if os.path.isdir("./vectorstore/" + dir)])
    indice_options = [m.replace(".index", "") for m in indice_options]
    indice = st.selectbox("Vector store", options=indice_options, index=0)
    num_embeddings_retornar = st.slider("Núm. embeddings retorno", min_value=1, max_value=10, value=3, step=1)

    # Adicionar espaço amtes do botão
    st.markdown("###")

    # Botão para limpar o histórico da conversa
    if st.sidebar.button("Limpar conversa"):
        st.session_state.mensagem = [{"role": "assistant", "content": mensagem_abertura}]
        st.session_state.historico = ConversationBufferWindowMemory(k=interacoes_memoria, memory_key="chat_history", input_key='question', output_key='answer', return_messages=True)

# Spinner enquanto o modelo é carregado inicialmente
with st.spinner("Carregando o modelo. Se for a primeira execução do modelo o mesmo será baixado e poderá demorar alguns minutos."):
    llm = carregar_llm(modelo, temperatura, tokens_contexto, top_k, top_p, repeat_penalty, num_threads)
    vectorstore = carregar_vectorstore()
    
# Histórico do LangChain que será utilizado para gerar o contexto para o LLM
if "historico" not in st.session_state:
    st.session_state.historico = ConversationBufferWindowMemory(k=interacoes_memoria, memory_key="chat_history", return_messages=True)

# Armazenar as perguntas e respostas
if "mensagem" not in st.session_state.keys():
    st.session_state.mensagem = [{"role": "assistant", "content": mensagem_abertura}]

# Exibir o histórico das mensagens
for message in st.session_state.mensagem:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Campo de pergunta 
if pergunta := st.chat_input("Digite sua pergunta"):
    st.session_state.mensagem.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        # Imprime na tela o prompt do usuário
        st.write(pergunta)

# Gerar nova resposta caso necessário
if st.session_state.mensagem[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Processando a resposta..."):
            resposta = obter_resposta(llm, vectorstore, pergunta, num_embeddings_retornar)
            
            resposta_completa = f"{resposta['answer']}   \n"
            resposta_completa += "   \n Documentos retornados para subsidiar a resposta:   \n   \n"

            for i, doc in enumerate(resposta['source_documents']):
                pcdt_fonte = Document.dict(doc)["metadata"]["source"]
                topico_fonte = Document.dict(doc)["metadata"]["topic"]
                resposta_completa += f"{i+1}. PCDT: {pcdt_fonte} - tópico: {topico_fonte}   \n"

            placeholder = st.empty()
            placeholder.markdown(resposta_completa)

            message = {"role": "assistant", "content": resposta_completa}
            st.session_state.mensagem.append(message)