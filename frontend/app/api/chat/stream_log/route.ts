// 基于 Next.js 的服务器环境设计的，主要功能是处理用户的 POST 请求，并利用 LangChain 框架和相关的 AI 模型生成响应。
import { NextRequest, NextResponse } from "next/server";

import type { Document } from "@langchain/core/documents";

import {
  Runnable,
  RunnableSequence,
  RunnableMap,
  RunnableBranch,
  RunnableLambda,
} from "@langchain/core/runnables";
import { HumanMessage, AIMessage, BaseMessage } from "@langchain/core/messages";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatFireworks } from "@langchain/community/chat_models/fireworks";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import weaviate, { ApiKey } from "weaviate-ts-client";
import { WeaviateStore } from "@langchain/weaviate";

export const runtime = "edge";

// 回答生成提示是这个 RAG 管道的最重要部分之一。没有一个好的提示，LLM 将无法（或严重受限）生成好的答案。提示在 RESPONSE_TEMPLATE 变量中定义。
// 这个模板包含了生成答案的所有必要信息，包括如何引用检索到的文档，如何使用 bullet points 和如何使用 unbiased 和 journalistic tone。
// 生成的答案应该是一个包含了检索到的文档的综合性和信息性的答案，不超过 80 个单词。
const RESPONSE_TEMPLATE = `You are an expert programmer and problem-solver, tasked to answer any question about Langchain.
Using the provided context, answer the user's question to the best of your ability using the resources provided.
Generate a comprehensive and informative answer (but no more than 80 words) for a given question based solely on the provided search results (URL and content).
You must only use information from the provided search results.
Use an unbiased and journalistic tone.
Combine search results together into a coherent answer.
Do not repeat text.
Cite search results using [\${{number}}] notation.
Only cite the most relevant results that answer the question accurately.
Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end.
If different results refer to different entities within the same name, write separate answers for each entity.
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.

You should use bullet points in your answer for readability
Put citations where they apply rather than putting them all at the end.

Anything between the following \`context\`  html blocks is retrieved from a knowledge bank, not part of the conversation with the user.

<context>
{context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer.
Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.`;

const REPHRASE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:`;

type RetrievalChainInput = {
  chat_history: string;
  question: string;
};

/*
* 主要功能是创建并返回一个Weaviate的检索器。
首先，它检查环境变量WEAVIATE_INDEX_NAME，WEAVIATE_API_KEY和WEAVIATE_URL是否已经设置。如果没有设置，它会抛出一个错误。
然后，它使用这些环境变量创建一个Weaviate客户端。
接着，它使用这个客户端和其他参数创建一个WeaviateStore实例。这个实例是从现有的索引创建的。
最后，它将WeaviateStore实例转换为检索器，并返回这个检索器。
*/
const getRetriever = async () => {
  if (
    !process.env.WEAVIATE_INDEX_NAME ||
    !process.env.WEAVIATE_API_KEY ||
    !process.env.WEAVIATE_URL
  ) {
    throw new Error(
      "WEAVIATE_INDEX_NAME, WEAVIATE_API_KEY and WEAVIATE_URL environment variables must be set",
    );
  }

  const client = weaviate.client({
    scheme: "https",
    host: process.env.WEAVIATE_URL,
    apiKey: new ApiKey(process.env.WEAVIATE_API_KEY),
  });
  const vectorstore = await WeaviateStore.fromExistingIndex(
    new OpenAIEmbeddings({}),
    {
      client,
      indexName: process.env.WEAVIATE_INDEX_NAME,
      textKey: "text",
      metadataKeys: ["source", "title"],
    },
  );
  return vectorstore.asRetriever({ k: 6 });
};

/*
* 函数的主要目的是创建一个检索链，这个链可以根据聊天历史的存在与否，选择不同的检索方式。
*/
const createRetrieverChain = (llm: BaseChatModel, retriever: Runnable) => {
  //创建一个提示模板
  const CONDENSE_QUESTION_PROMPT =
    PromptTemplate.fromTemplate(REPHRASE_TEMPLATE);
  //创建一个检索链
  const condenseQuestionChain = RunnableSequence.from([
    CONDENSE_QUESTION_PROMPT,
    llm,
    new StringOutputParser(),
  ]).withConfig({
    runName: "CondenseQuestion",
  });
  //创建一个检查函数，检查聊天历史是否存在
  const hasHistoryCheckFn = RunnableLambda.from(
    (input: RetrievalChainInput) => input.chat_history.length > 0,
  ).withConfig({ runName: "HasChatHistoryCheck" });
  //创建一个分支，根据聊天历史的存在与否，选择不同的检索方式
  const conversationChain = condenseQuestionChain.pipe(retriever).withConfig({
    runName: "RetrievalChainWithHistory",
  });
  //创建一个基本的检索链
  const basicRetrievalChain = RunnableLambda.from(
    (input: RetrievalChainInput) => input.question,
  )
    .withConfig({
      runName: "Itemgetter:question",
    })
    .pipe(retriever)
    .withConfig({ runName: "RetrievalChainWithNoHistory" });
  // 返回一个RunnableBranch实例，这个实例包含了两个检索链，根据聊天历史的存在与否，选择不同的检索方式
  return RunnableBranch.from([
    [hasHistoryCheckFn, conversationChain],
    basicRetrievalChain,
  ]).withConfig({
    runName: "FindDocs",
  });
};
// 格式化文档
const formatDocs = (docs: Document[]) => {
  return docs
    .map((doc, i) => `<doc id='${i}'>${doc.pageContent}</doc>`)
    .join("\n");
};
// 格式化聊天历史
const formatChatHistoryAsString = (history: BaseMessage[]) => {
  return history
    .map((message) => `${message._getType()}: ${message.content}`)
    .join("\n");
};
// 序列化历史
// 函数的主要功能是将input中的chat_history转换为一个新的数组convertedChatHistory，这个数组中的元素是HumanMessage或AIMessage的实例
const serializeHistory = (input: any) => {
  const chatHistory = input.chat_history || [];
  const convertedChatHistory = [];
  for (const message of chatHistory) {
    if (message.human !== undefined) {
      convertedChatHistory.push(new HumanMessage({ content: message.human }));
    }
    if (message["ai"] !== undefined) {
      convertedChatHistory.push(new AIMessage({ content: message.ai }));
    }
  }
  return convertedChatHistory;
};
// 创建一个检索链
// 函数的主要功能是创建一个运行序列，这个序列包含了处理聊天历史、检索文档和生成响应的所有步骤。

const createChain = (llm: BaseChatModel, retriever: Runnable) => {
  //创建一个检索链
  const retrieverChain = createRetrieverChain(llm, retriever);
  //创建一个上下文
  const context = RunnableMap.from({
    context: RunnableSequence.from([
      ({ question, chat_history }) => ({
        question,
        chat_history: formatChatHistoryAsString(chat_history),
      }),
      retrieverChain,
      RunnableLambda.from(formatDocs).withConfig({
        runName: "FormatDocumentChunks",
      }),
    ]),
    question: RunnableLambda.from(
      (input: RetrievalChainInput) => input.question,
    ).withConfig({
      runName: "Itemgetter:question",
    }),
    chat_history: RunnableLambda.from(
      (input: RetrievalChainInput) => input.chat_history,
    ).withConfig({
      runName: "Itemgetter:chat_history",
    }),
  }).withConfig({ tags: ["RetrieveDocs"] });
  //创建一个提示模板
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", RESPONSE_TEMPLATE],
    new MessagesPlaceholder("chat_history"),
    ["human", "{question}"],
  ]);
//创建一个运行序列，这个序列首先运行提示模板，然后运行llm，最后解析输出为字符串。
  const responseSynthesizerChain = RunnableSequence.from([
    prompt,
    llm,
    new StringOutputParser(),
  ]).withConfig({
    tags: ["GenerateResponse"],
  });
  return RunnableSequence.from([
    {
      question: RunnableLambda.from(
        (input: RetrievalChainInput) => input.question,
      ).withConfig({
        runName: "Itemgetter:question",
      }),
      chat_history: RunnableLambda.from(serializeHistory).withConfig({
        runName: "SerializeHistory",
      }),
    },
    context,
    responseSynthesizerChain,
  ]);
};

/*
· 该函数的主要功能是根据请求中的输入和配置，生成一个检索链，并将检索链的输出流式传递给前端。
 这个函数可能是一个Express.js或者类似框架的路由处理器，用于处理POST请求。
*/
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const input = body.input;
    const config = body.config;

    // 创建一个ChatOpenAI或ChatFireworks实例，并将其存储在llm变量中
    let llm;
    if (config.configurable.llm === "openai_gpt_3_5_turbo") {
      llm = new ChatOpenAI({
        modelName: "gpt-3.5-turbo-1106",
        temperature: 0,
      });
    } else if (config.configurable.llm === "fireworks_mixtral") {
      llm = new ChatFireworks({
        modelName: "accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature: 0,
      });
    } else {
      throw new Error(
        "Invalid LLM option passed. Must be 'openai' or 'mixtral'. Received: " +
          config.llm,
      );
    }
    // 获取一个检索器 
    const retriever = await getRetriever();
    // 并使用llm和检索器创建一个答案链
    const answerChain = createChain(llm, retriever);

    /**
     * Narrows streamed log output down to final output and the FindDocs tagged chain to
     * selectively stream back sources.
     *
     * You can use .stream() to create a ReadableStream with just the final output which
     * you can pass directly to the Response as well:
     * https://js.langchain.com/docs/expression_language/interface#stream
     */
    // 将检索链以streamLog的形式输出
    const stream = answerChain.streamLog(input, config, {
      includeNames: body.includeNames,
    });

    // Only return a selection of output to the frontend
    const textEncoder = new TextEncoder();
    // 将答案链的日志输出编码为文本
    const clientStream = new ReadableStream({
      async start(controller) {
        for await (const chunk of stream) {
          controller.enqueue(
            textEncoder.encode(
              "event: data\ndata: " + JSON.stringify(chunk) + "\n\n",
            ),
          );
        }
        controller.enqueue(textEncoder.encode("event: end\n\n"));
        controller.close();
      },
    });
    // 返回一个包含流的响应，状态码为200，内容类型为text/event-stream。
    return new Response(clientStream, {
      headers: { "Content-Type": "text/event-stream" },
    });
  } catch (e: any) {
    // 如果在以上过程中发生任何错误，它会捕获这个错误，并返回一个包含错误信息的JSON响应，状态码为500。
    console.error(e);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
