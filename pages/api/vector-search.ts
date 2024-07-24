import type { NextRequest } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { codeBlock, oneLine } from 'common-tags'
import GPT3Tokenizer from 'gpt3-tokenizer'
// import {
//   Configuration,
//   OpenAIApi,
//   CreateModerationResponse,
//   CreateEmbeddingResponse,
//   ChatCompletionRequestMessage,
// } from 'openai-edge'
import { OpenAIStream, StreamingTextResponse } from 'ai'
import { ApplicationError, UserError } from '@/lib/errors'
import { OpenAI } from 'openai'

const openAiKey = process.env.OPENAI_KEY
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY

// const config = new Configuration({
//   apiKey: openAiKey,
// })
// const openai = new OpenAIApi(config)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_KEY,
  // baseURL: 'http://10.128.138.175:11434/v1',
  baseURL: 'http://10.128.138.175:8080/v1',
})

export const runtime = 'edge'

export default async function handler(req: NextRequest) {
  try {
    if (!openAiKey) {
      throw new ApplicationError('Missing environment variable OPENAI_KEY')
    }

    if (!supabaseUrl) {
      throw new ApplicationError('Missing environment variable SUPABASE_URL')
    }

    if (!supabaseServiceKey) {
      throw new ApplicationError('Missing environment variable SUPABASE_SERVICE_ROLE_KEY')
    }

    const requestData = await req.json()

    if (!requestData) {
      throw new UserError('Missing request data')
    }

    const { prompt: query } = requestData

    if (!query) {
      throw new UserError('Missing query in request data')
    }

    const supabaseClient = createClient(supabaseUrl, supabaseServiceKey)

    // Moderate the content to comply with OpenAI T&C
    const sanitizedQuery = query.trim().replace(/[\r\n]+/g, ' ')
    // const moderationResponse: CreateModerationResponse = await openai
    //   .createModeration({ input: sanitizedQuery })
    //   .then((res) => res.json())

    // const [results] = moderationResponse.results

    // if (results.flagged) {
    //   throw new UserError('Flagged content', {
    //     flagged: true,
    //     categories: results.categories,
    //   })
    // }

    // Create embedding from query
    // const embeddingResponse = await openai.createEmbedding({
    //   model: 'text-embedding-ada-002',
    //   input: sanitizedQuery.replaceAll('\n', ' '),
    // })

    const embeddingResponse = await openai.embeddings.create({
      model: 'nomic-embed-text:latest',
      input: sanitizedQuery,
    })

    // console.log(`embeddingResponse: ${JSON.stringify(embeddingResponse)}`)

    // if (embeddingResponse.status !== 200) {
    //   throw new ApplicationError('Failed to create embedding for question', embeddingResponse)
    // }

    // const {
    //   data: [{ embedding }],
    // }: CreateEmbeddingResponse = await embeddingResponse
    const [responseData] = embeddingResponse.data
    console.log(`responseData: ${JSON.stringify(responseData.embedding)}`)

    // console.log(`embedding: ${embedding}`)

    const { error: matchError, data: pageSections } = await supabaseClient.rpc(
      'match_page_sections',
      {
        embedding: responseData.embedding,
        match_threshold: 0.20,
        match_count: 10,
        min_content_length: 50,
      })

    console.log(`pageSections: ${JSON.stringify(pageSections)}`)

    if (matchError) {
      throw new ApplicationError('Failed to match page sections', matchError)
    }

    const tokenizer = new GPT3Tokenizer({ type: 'gpt3' })
    let tokenCount = 0
    let contextText = ''

    for (let i = 0; i < pageSections.length; i++) {
      const pageSection = pageSections[i]
      const content = pageSection.content
      const encoded = tokenizer.encode(content)
      tokenCount += encoded.text.length

      if (tokenCount >= 1500) {
        break
      }

      contextText += `${content.trim()}\n---\n`
    }

    console.log(`contextText: ${contextText}`)
    console.log(`sanitizedQuery: ${sanitizedQuery}`)

    const prompt = codeBlock`
      ${oneLine`
        You're an AI assistant who answers questions about documents and related code snippets.
      
        You're a chat bot, so keep your replies succinct and conversational.

        You're only allowed to use the documents below to answer the question.

        If the question isn't related to these documents, say:
        "Sorry, I couldn't find any information from the documents."

        If the information isn't available in below documents, say:
        "Sorry, I couldn't find any information from the documents."

        Do not go off topic.
      `}

      Documents:
      ${contextText}

      Question: """
      ${query}
      """

      Answer as markdown (including related code snippets if available):
    `

    const chatMessage: OpenAI.Chat.Completions.ChatCompletionMessageParam = {
      role: 'user',
      content: prompt,
    }

    const openai_ollama = new OpenAI({
      apiKey: 'ollama',
      // baseURL: 'http://10.128.138.175:8000/v1/',
      baseURL: 'http://10.128.138.175:11434/v1/',
      // baseURL: "http://host.docker.internal:11434/v1/",
    });

    const response = await openai_ollama.chat.completions.create({
      model: 'cira-dpo-llama2:latest',
      messages: [chatMessage],
      max_tokens: 512,
      temperature: 0,
      stream: true,
    })

    // if (!response.ok) {
    //   const error = await response.json()
    //   throw new ApplicationError('Failed to generate completion', error)
    // }

    // Transform the response into a readable stream
    const stream = OpenAIStream(response)
    console.log(`stream: ${stream}`)

    // Return a StreamingTextResponse, which can be consumed by the client
    return new StreamingTextResponse(stream)
  } catch (err: unknown) {
    if (err instanceof UserError) {
      return new Response(
        JSON.stringify({
          error: err.message,
          data: err.data,
        }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    } else if (err instanceof ApplicationError) {
      // Print out application errors with their additional data
      console.error(`${err.message}: ${JSON.stringify(err.data)}`)
    } else {
      // Print out unexpected errors as is to help with debugging
      console.error(err)
    }

    // TODO: include more response info in debug environments
    return new Response(
      JSON.stringify({
        error: 'There was an error processing your request',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}
