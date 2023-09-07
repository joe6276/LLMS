const {OpenAIEmbeddings}  =require("langchain/embeddings/openai")
const {CharacterTextSplitter} =require( "langchain/text_splitter")
const { FaissStore } =require("langchain/vectorstores/faiss");
const { PDFLoader } =require( "langchain/document_loaders/fs/pdf")
const {OpenAI} =require("langchain/llms/openai")
const  { loadQAStuffChain } =require("langchain/chains");
const openAIApiKey= "API_KEY"


async function ReadFile(){
    //read pdf

    const loader = new PDFLoader("Almanack.pdf",{
        splitPages:false
    })

    const docs = await loader.load();

    // console.log(docs)

    let raw_text=""
    docs.forEach(page=>raw_text+=page.pageContent)

    const textSplitter= new CharacterTextSplitter({
        separator:'\n',
        chunkSize:800,
        chunkOverlap:200,
        lengthFunction:(text)=>text.length
    })

    const texts = await textSplitter.splitText(raw_text)
    // console.log(texts.length)

    const vectorStore= await FaissStore.fromTexts(texts,{}, new OpenAIEmbeddings({openAIApiKey}))
    // console.log(vectorStore)

    const query = "Tell me 3 paragraphs to summarize this document"

    const result= await vectorStore.similaritySearch(query,2)

    // console.log(result)

    const llm = new OpenAI({openAIApiKey, verbose:true})// add ==>verbose:true (if you want to see the steps taken)
    const chains=loadQAStuffChain(llm)

    const answer = await chains.call({
        input_documents:result,
        question:query
    })

    console.log(answer.text)


}


ReadFile()
