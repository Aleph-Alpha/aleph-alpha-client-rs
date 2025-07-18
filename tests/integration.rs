use std::{fs::File, io::BufReader};

use aleph_alpha_client::{
    cosine_similarity, ChatEvent, ChatSampling, Client, CompletionEvent, Error, Granularity, How,
    ImageScore, ItemExplanation, Logprobs, Message, Modality, Prompt, PromptGranularity, Sampling,
    SemanticRepresentation, Stopping, Task, TaskBatchSemanticEmbedding, TaskChat, TaskCompletion,
    TaskDetokenization, TaskExplanation, TaskSemanticEmbedding,
    TaskSemanticEmbeddingWithInstruction, TaskTokenization, TextScore, TraceContext, Usage,
};
use dotenvy::dotenv;
use futures_util::StreamExt;
use image::ImageFormat;
use std::sync::LazyLock;

fn pharia_ai_token() -> &'static str {
    static PHARIA_AI_TOKEN: LazyLock<String> = LazyLock::new(|| {
        drop(dotenv());
        std::env::var("PHARIA_AI_TOKEN")
            .expect("PHARIA_AI_TOKEN environment variable must be specified to run tests.")
    });
    &PHARIA_AI_TOKEN
}

fn inference_url() -> &'static str {
    static INFERENCE_URL: LazyLock<String> = LazyLock::new(|| {
        drop(dotenv());
        std::env::var("INFERENCE_URL")
            .expect("INFERENCE_URL environment variable must be specified to run tests.")
    });
    &INFERENCE_URL
}

#[tokio::test]
async fn chat_with_pharia_1_7b_base() {
    // When
    let message = Message::user("Question");
    let task = TaskChat::with_message(message);

    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client.chat(&task, model, &How::default()).await.unwrap();

    // Then
    assert!(!response.message.content.is_empty())
}

#[tokio::test]
async fn completion_with_luminous_base() {
    // When
    let task = TaskCompletion::from_text("Hello").with_maximum_tokens(1);

    let model = "luminous-base";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    assert!(!response.completion.is_empty())
}

#[tokio::test]
async fn raw_completion_includes_python_tag() {
    // When
    let task = TaskCompletion::from_text(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython<|eot_id|><|start_header_id|>user<|end_header_id|>

Write code to check if number is prime, use that to see if the number 7 is prime<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    )
    .with_maximum_tokens(30)
    .with_special_tokens();

    let model = "llama-3.1-8b-instruct";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();
    assert!(response.completion.trim().starts_with("<|python_tag|>"));
}

#[tokio::test]
async fn request_authentication_has_priority() {
    let bad_pharia_ai_token = "DUMMY";
    let task = TaskCompletion::from_text("Hello").with_maximum_tokens(1);

    let model = "luminous-base";
    let client = Client::with_auth(inference_url(), bad_pharia_ai_token).unwrap();
    let response = client
        .output_of(
            &task.with_model(model),
            &How {
                api_token: Some(pharia_ai_token().to_owned()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    eprintln!("{}", response.completion);

    // Then
    assert!(!response.completion.is_empty())
}

#[tokio::test]
async fn authentication_only_per_request() {
    // Given
    let model = "luminous-base";
    let task = TaskCompletion::from_text("Hello").with_maximum_tokens(1);

    // When
    let client = Client::new(inference_url().to_owned(), None).unwrap();
    let response = client
        .output_of(
            &task.with_model(model),
            &How {
                api_token: Some(pharia_ai_token().to_owned()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Then there is some successful completion
    assert!(!response.completion.is_empty())
}

#[should_panic = "API token needs to be set on client construction or per request"]
#[tokio::test]
async fn must_panic_if_authentication_is_missing() {
    // Given
    let model = "luminous-base";
    let task = TaskCompletion::from_text("Hello").with_maximum_tokens(1);

    // When
    let client = Client::new(inference_url().to_owned(), None).unwrap();
    client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then the client panics on invocation
}

#[tokio::test]
async fn semantic_search_with_luminous_base() {
    // Given
    let robot_fact = Prompt::from_text(
        "A robot is a machine—especially one programmable by a computer—capable of carrying out a \
        complex series of actions automatically.",
    );
    let pizza_fact = Prompt::from_text(
        "Pizza (Italian: [ˈpittsa], Neapolitan: [ˈpittsə]) is a dish of Italian origin consisting \
        of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, \
        and often various other ingredients (such as various types of sausage, anchovies, \
        mushrooms, onions, olives, vegetables, meat, ham, etc.), which is then baked at a high \
        temperature, traditionally in a wood-fired oven.",
    );
    let query = Prompt::from_text("What is Pizza?");
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let robot_embedding_task = TaskSemanticEmbedding {
        prompt: robot_fact,
        representation: SemanticRepresentation::Document,
        compress_to_size: Some(128),
    };
    let robot_embedding = client
        .semantic_embedding(&robot_embedding_task, &How::default())
        .await
        .unwrap()
        .embedding;

    let pizza_embedding_task = TaskSemanticEmbedding {
        prompt: pizza_fact,
        representation: SemanticRepresentation::Document,
        compress_to_size: Some(128),
    };
    let pizza_embedding = client
        .semantic_embedding(&pizza_embedding_task, &How::default())
        .await
        .unwrap()
        .embedding;

    let query_embedding_task = TaskSemanticEmbedding {
        prompt: query,
        representation: SemanticRepresentation::Query,
        compress_to_size: Some(128),
    };
    let query_embedding = client
        .semantic_embedding(&query_embedding_task, &How::default())
        .await
        .unwrap()
        .embedding;
    let similarity_pizza = cosine_similarity(&query_embedding, &pizza_embedding);
    println!("similarity pizza: {similarity_pizza}");
    let similarity_robot = cosine_similarity(&query_embedding, &robot_embedding);
    println!("similarity robot: {similarity_robot}");

    // Then

    // The fact about pizza should be more relevant to the "What is Pizza?" question than a fact
    // about robots.
    assert!(similarity_pizza > similarity_robot);
}

#[tokio::test]
async fn complete_structured_prompt() {
    // Given
    let prompt = "Bot: Hello user!\nUser: Hello Bot, how are you doing?\nBot:";
    let stop_sequences = ["User:"];

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text(prompt),
        stopping: Stopping {
            maximum_tokens: Some(64),
            stop_sequences: &stop_sequences[..],
        },
        sampling: Sampling::MOST_LIKELY,
        special_tokens: false,
        logprobs: Logprobs::No,
        echo: false,
    };
    let model = "luminous-base";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    eprintln!("{}", response.completion);
    assert!(!response.completion.is_empty());
    assert!(!response.completion.contains("User:"));
}

#[tokio::test]
async fn maximum_tokens_none_request() {
    // Given
    let prompt = "Bot: Hello user!\nUser: Hello Bot, how are you doing?\nBot:";
    let var_name = Stopping {
        maximum_tokens: None,
        stop_sequences: &["User"],
    };
    let stopping = var_name;

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text(prompt),
        stopping,
        sampling: Sampling::MOST_LIKELY,
        special_tokens: false,
        logprobs: Logprobs::No,
        echo: false,
    };
    let model = "luminous-base";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    assert!(!response.completion.is_empty());
    assert_eq!(response.completion, " I am doing fine, how are you?\n");
}

#[tokio::test]
async fn echo_prompt_request_without_logprobs() {
    // Given
    let prompt = " An apple a day";
    let stopping = Stopping::from_maximum_tokens(10);

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text(prompt),
        stopping,
        sampling: Sampling::MOST_LIKELY,
        special_tokens: false,
        logprobs: Logprobs::No,
        echo: true,
    };
    let model = "luminous-base";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    assert!(response.completion.starts_with(prompt));
}

#[tokio::test]
async fn echo_prompt_request_with_sampled_logprobs() {
    // Given
    let prompt = "apple";
    let stopping = Stopping::from_maximum_tokens(1);

    // When requesting a completion with sampled logprobs
    let task = TaskCompletion {
        prompt: Prompt::from_text(prompt),
        stopping,
        sampling: Sampling::MOST_LIKELY,
        special_tokens: false,
        logprobs: Logprobs::Sampled,
        echo: true,
    };
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then all the top logprobs are empty
    assert_eq!(response.logprobs.len(), 3);
    assert_eq!(response.logprobs[0].top.len(), 0);
    assert_eq!(response.logprobs[1].top.len(), 0);
    assert_eq!(response.logprobs[2].top.len(), 0);

    // And the logprob for only the first token is NAN
    assert!(response.logprobs[0].sampled.logprob.is_nan());

    // And the logprob for the second and third token are not NAN
    assert!(!response.logprobs[1].sampled.logprob.is_nan());
    assert!(!response.logprobs[2].sampled.logprob.is_nan());
}

#[tokio::test]
async fn echo_prompt_request_with_logprobs() {
    // Given
    let prompt = "apple";
    let stopping = Stopping::from_maximum_tokens(1);

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text(prompt),
        stopping,
        sampling: Sampling::MOST_LIKELY,
        special_tokens: false,
        logprobs: Logprobs::Top(3),
        echo: true,
    };
    let model = "luminous-base";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then we do not get logprobs for the first token, but for the second one
    assert_eq!(response.logprobs.len(), 2);
    assert_eq!(response.logprobs[0].top.len(), 0);
    assert_eq!(response.logprobs[1].top.len(), 3);
}

#[tokio::test]
async fn explain_request() {
    // Given
    let input = "Hello World!";
    let num_input_sentences = 1; // keep in sync with input
    let task = TaskExplanation {
        prompt: Prompt::from_text(input),
        target: " How is it going?",
        granularity: Granularity::default().with_prompt_granularity(PromptGranularity::Sentence),
    };
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let response = client
        .explanation(
            &task,
            "luminous-base",
            &How {
                be_nice: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Then
    assert_eq!(response.items.len(), 2); // 1 text + 1 target
    assert_eq!(text_scores(&response.items[0]).len(), num_input_sentences)
}

#[tokio::test]
async fn explain_request_with_auto_granularity() {
    // Given
    let input = "Hello World!";
    let num_input_tokens = 3; // keep in sync with input
    let task = TaskExplanation {
        prompt: Prompt::from_text(input),
        target: " How is it going?",
        granularity: Granularity::default(),
    };
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let response = client
        .explanation(
            &task,
            "luminous-base",
            &How {
                be_nice: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Then
    assert_eq!(text_scores(&response.items[0]).len(), num_input_tokens)
}

#[tokio::test]
async fn explain_request_with_image_modality() {
    // Given
    let input = Prompt::from_vec(vec![
        Modality::from_image_path("tests/cat-chat-1641458.jpg").unwrap(),
        Modality::from_text("A picture of "),
    ]);
    let num_input_images = 1; // keep in sync with input
    let task = TaskExplanation {
        prompt: input,
        target: " a cat.",
        granularity: Granularity::default().with_prompt_granularity(PromptGranularity::Paragraph),
    };
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let response = client
        .explanation(
            &task,
            "luminous-base",
            &How {
                be_nice: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Then
    assert_eq!(image_scores(&response.items[0]).len(), num_input_images)
}

fn text_scores(item: &ItemExplanation) -> Vec<TextScore> {
    match item {
        ItemExplanation::Text { scores } => scores.to_vec(),
        ItemExplanation::Target { scores } => scores.to_vec(),
        ItemExplanation::Image { .. } => Vec::new(),
    }
}

fn image_scores(item: &ItemExplanation) -> Vec<ImageScore> {
    match item {
        ItemExplanation::Text { .. } => Vec::new(),
        ItemExplanation::Target { .. } => Vec::new(),
        ItemExplanation::Image { scores } => scores.to_vec(),
    }
}

#[tokio::test]
async fn describe_image_starting_from_a_path() {
    // Given
    let path_to_image = "tests/cat-chat-1641458.jpg";

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_vec(vec![
            Modality::from_image_path(path_to_image).unwrap(),
            Modality::from_text("A picture of "),
        ]),
        stopping: Stopping::from_maximum_tokens(10),
        sampling: Sampling::MOST_LIKELY,
        special_tokens: false,
        logprobs: Logprobs::No,
        echo: false,
    };
    let model = "luminous-base";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    eprintln!("{}", response.completion);
    assert!(response.completion.contains("cat"))
}

#[tokio::test]
async fn describe_image_starting_from_a_dyn_image() {
    // Given
    let path_to_image = "tests/cat-chat-1641458.jpg";
    let file = BufReader::new(File::open(path_to_image).unwrap());
    let format = ImageFormat::from_path(path_to_image).unwrap();
    let image = image::load(file, format).unwrap();

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_vec(vec![
            Modality::from_image(&image).unwrap(),
            Modality::from_text("A picture of "),
        ]),
        stopping: Stopping::from_maximum_tokens(10),
        sampling: Sampling::MOST_LIKELY,
        special_tokens: false,
        logprobs: Logprobs::No,
        echo: false,
    };
    let model = "luminous-base";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    eprintln!("{}", response.completion);
    assert!(response.completion.contains("cat"))
}

#[tokio::test]
async fn batch_semantic_embed_with_luminous_base() {
    // Given
    let robot_fact = Prompt::from_text(
        "A robot is a machine—especially one programmable by a computer—capable of carrying out a \
        complex series of actions automatically.",
    );
    let pizza_fact = Prompt::from_text(
        "Pizza (Italian: [ˈpittsa], Neapolitan: [ˈpittsə]) is a dish of Italian origin consisting \
        of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, \
        and often various other ingredients (such as various types of sausage, anchovies, \
        mushrooms, onions, olives, vegetables, meat, ham, etc.), which is then baked at a high \
        temperature, traditionally in a wood-fired oven.",
    );

    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let embedding_task = TaskBatchSemanticEmbedding {
        prompts: vec![robot_fact, pizza_fact],
        representation: SemanticRepresentation::Document,
        compress_to_size: Some(128),
    };

    let embeddings = client
        .batch_semantic_embedding(&embedding_task, &How::default())
        .await
        .unwrap()
        .embeddings;

    // Then
    // There should be 2 embeddings
    assert_eq!(embeddings.len(), 2);
}

#[tokio::test]
async fn semantic_embed_with_instruction_with_luminous_base() {
    // Given
    let robot_fact = Prompt::from_text(
        "A robot is a machine—especially one programmable by a computer—capable of carrying out a \
        complex series of actions automatically.",
    );
    let pizza_fact = Prompt::from_text(
        "Pizza (Italian: [ˈpittsa], Neapolitan: [ˈpittsə]) is a dish of Italian origin consisting \
        of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, \
        and often various other ingredients (such as various types of sausage, anchovies, \
        mushrooms, onions, olives, vegetables, meat, ham, etc.), which is then baked at a high \
        temperature, traditionally in a wood-fired oven.",
    );
    let query = Prompt::from_text("What is Pizza?");
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let robot_embedding_task = TaskSemanticEmbeddingWithInstruction {
        instruction: "Embed this fact",
        prompt: robot_fact,
        normalize: None,
    };
    let robot_embedding = client
        .semantic_embedding_with_instruction(&robot_embedding_task, &How::default())
        .await
        .unwrap()
        .embedding;

    let pizza_embedding_task = TaskSemanticEmbeddingWithInstruction {
        instruction: "Embed this fact",
        prompt: pizza_fact,
        normalize: None,
    };
    let pizza_embedding = client
        .semantic_embedding_with_instruction(&pizza_embedding_task, &How::default())
        .await
        .unwrap()
        .embedding;

    let query_embedding_task = TaskSemanticEmbeddingWithInstruction {
        instruction: "Embed this question about facts",
        prompt: query,
        normalize: None,
    };
    let query_embedding = client
        .semantic_embedding_with_instruction(&query_embedding_task, &How::default())
        .await
        .unwrap()
        .embedding;

    let similarity_pizza = cosine_similarity(&query_embedding, &pizza_embedding);
    println!("similarity pizza: {similarity_pizza}");
    let similarity_robot = cosine_similarity(&query_embedding, &robot_embedding);
    println!("similarity robot: {similarity_robot}");

    // Then

    // The fact about pizza should be more relevant to the "What is Pizza?" question than a fact
    // about robots.
    assert!(similarity_pizza > similarity_robot);
}

#[tokio::test]
async fn tokenization_with_luminous_base() {
    // Given
    let input = "Hello, World!";

    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let task1 = TaskTokenization::new(input, false, true);
    let task2 = TaskTokenization::new(input, true, false);

    let response1 = client
        .tokenize(&task1, "luminous-base", &How::default())
        .await
        .unwrap();

    let response2 = client
        .tokenize(&task2, "luminous-base", &How::default())
        .await
        .unwrap();

    // Then
    assert_eq!(response1.tokens, None);
    assert_eq!(response1.token_ids, Some(vec![49222, 15, 5390, 4]));

    assert_eq!(response2.token_ids, None);
    assert_eq!(
        response2.tokens,
        Some(
            vec!["ĠHello", ",", "ĠWorld", "!"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        )
    );
}

#[tokio::test]
async fn detokenization_with_luminous_base() {
    // Given
    let input = vec![49222, 15, 5390, 4];

    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let task = TaskDetokenization { token_ids: &input };

    let response = client
        .detokenize(&task, "luminous-base", &How::default())
        .await
        .unwrap();

    // Then
    assert!(response.result.contains("Hello, World!"));
}

#[tokio::test]
async fn fetch_tokenizer_for_pharia_1_llm_7b() {
    // Given
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let tokenizer = client
        .tokenizer_by_model("pharia-1-llm-7b-control", None, None)
        .await
        .unwrap();

    // Then
    assert_eq!(128_000, tokenizer.get_vocab_size(true));
}

#[tokio::test]
async fn stream_completion() {
    // Given a streaming completion task
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let task = TaskCompletion::from_text("").with_maximum_tokens(7);

    // When the events are streamed and collected
    let mut stream = client
        .stream_completion(&task, "luminous-base", &How::default())
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Some(Ok(event)) = stream.next().await {
        events.push(event);
    }

    // Then there are at least one chunk, one summary and one completion summary
    assert!(events.len() >= 3);
    assert!(matches!(
        events[events.len() - 3],
        CompletionEvent::Delta { .. }
    ));
    assert!(matches!(
        events[events.len() - 2],
        CompletionEvent::Finished { .. }
    ));
    assert!(matches!(
        events[events.len() - 1],
        CompletionEvent::Summary { .. }
    ));
}

#[tokio::test]
async fn stream_completion_special_tokens() {
    // Given a streaming completion task
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let task = TaskCompletion::from_text(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>

An apple a day<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    )
    .with_maximum_tokens(10)
    .with_special_tokens();

    // When the events are streamed and collected
    let mut stream = client
        .stream_completion(&task, "pharia-1-llm-7b-control", &How::default())
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Some(Ok(event)) = stream.next().await {
        events.push(event);
    }

    // Then there are at least one chunk, one summary and one completion summary
    assert_eq!(
        events,
        vec![
            CompletionEvent::Delta {
                completion: " \n\n Keeps the doctor away<|endoftext|>".to_owned(),
                logprobs: vec![]
            },
            CompletionEvent::Finished {
                reason: "end_of_text".to_owned()
            },
            CompletionEvent::Summary {
                usage: Usage {
                    prompt_tokens: 16,
                    completion_tokens: 9,
                }
            }
        ]
    );
}

#[tokio::test]
async fn stream_completion_sampled_logprobs() {
    // Given a streaming completion task
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let task = TaskCompletion::from_text(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>

An apple a day<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    .with_maximum_tokens(2)
    .with_logprobs(Logprobs::Sampled);

    // When the events are streamed and collected
    let mut stream = client
        .stream_completion(&task, "pharia-1-llm-7b-control", &How::default())
        .await
        .unwrap();

    let event = stream.next().await.unwrap().unwrap();

    let CompletionEvent::Delta { logprobs, .. } = event else {
        panic!("Unexpected event type");
    };

    assert_eq!(logprobs[0].sampled.token_as_str().unwrap(), " Keep");
    assert_eq!(logprobs[1].sampled.token_as_str().unwrap(), "s");
}

#[tokio::test]
async fn stream_chat_with_pharia_1_llm_7b() {
    // Given a streaming completion task
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let message = Message::user("An apple a day");
    let task = TaskChat::with_messages(vec![message]).with_maximum_tokens(7);

    // When the events are streamed and collected
    let stream = client
        .stream_chat(&task, "pharia-1-llm-7b-control", &How::default())
        .await
        .unwrap();

    let events = stream.collect::<Vec<_>>().await;

    // Then we receive three events, with the last one being a finished event
    assert_eq!(events.len(), 4);
    assert_eq!(
        events[0].as_ref().unwrap(),
        &ChatEvent::MessageStart {
            role: "assistant".to_owned()
        }
    );
    assert!(matches!(events[1], Ok(ChatEvent::MessageDelta { .. })));
    assert_eq!(
        events[2].as_ref().unwrap(),
        &ChatEvent::MessageEnd {
            stop_reason: "stop".to_owned()
        }
    );
    assert!(matches!(events[3], Ok(ChatEvent::Summary { .. })));
}

#[tokio::test]
async fn stream_chat_with_logprobs() {
    // Given a streaming completion task
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let message = Message::user("An apple a day");
    let task = TaskChat::with_messages(vec![message])
        .with_maximum_tokens(2)
        .with_logprobs(Logprobs::Sampled);

    // When the events are streamed and collected
    let mut stream = client
        .stream_chat(&task, "pharia-1-llm-7b-control", &How::default())
        .await
        .unwrap();

    // Role
    stream.next().await;
    // Content
    let event = stream.next().await.unwrap().unwrap();
    let ChatEvent::MessageDelta { logprobs, .. } = event else {
        panic!("Unexpected event type")
    };
    assert_eq!(logprobs[0].sampled.token_as_str().unwrap(), " Keep");
    assert_eq!(logprobs[1].sampled.token_as_str().unwrap(), "s");
}

#[tokio::test]
async fn frequency_penalty_request() {
    // Given a high negative frequency penalty
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let message = Message::user("Haiku about oat milk!");
    let sampling = ChatSampling {
        frequency_penalty: Some(-10.0),
        ..Default::default()
    };
    let stopping = Stopping::from_maximum_tokens(20);
    let task = TaskChat {
        messages: vec![message],
        stopping,
        sampling,
        logprobs: Logprobs::No,
    };

    // When the response is requested
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then we get a response with the word "white" appearing more than 10 times
    assert!(!response.message.content.is_empty());
    let count = response
        .message
        .content
        .to_lowercase()
        .split_whitespace()
        .filter(|word| *word == "oat")
        .count();
    assert!(count > 5);
}

#[tokio::test]
async fn presence_penalty_request() {
    // Given a high negative presence penalty
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let message = Message::user("Haiku about oat milk!");
    let sampling = ChatSampling {
        presence_penalty: Some(-10.0),
        ..Default::default()
    };
    let stopping = Stopping::from_maximum_tokens(20);
    let task = TaskChat {
        messages: vec![message],
        stopping,
        sampling,
        logprobs: Logprobs::No,
    };

    // When the response is requested
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then we get a response with the word "white" appearing more than 10 times
    assert!(!response.message.content.is_empty());
    let count = response
        .message
        .content
        .to_lowercase()
        .split_whitespace()
        .filter(|word| *word == "oat")
        .count();
    assert!(count > 5);
}

#[tokio::test]
async fn stop_sequences_request() {
    // Given a stop sequence
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let message = Message::user("An apple a day");

    let stopping = Stopping {
        stop_sequences: &["doctor"],
        maximum_tokens: None,
    };
    let task = TaskChat {
        messages: vec![message],
        stopping,
        sampling: ChatSampling::MOST_LIKELY,
        logprobs: Logprobs::No,
    };

    // When the response is requested
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    assert_eq!(response.finish_reason, "stop");
}

#[tokio::test]
async fn show_logprobs_sampled_chat() {
    // Given
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let message = Message::user("An apple a day");

    // When
    let task = TaskChat {
        messages: vec![message],
        stopping: Stopping::from_maximum_tokens(2),
        sampling: ChatSampling::MOST_LIKELY,
        logprobs: Logprobs::Sampled,
    };

    let response = client.chat(&task, model, &How::default()).await.unwrap();

    // Then
    assert_eq!(response.logprobs.len(), 2);
    assert_eq!(
        response.logprobs[0].sampled.token_as_str().unwrap(),
        " Keep"
    );
    assert_eq!(response.logprobs[1].sampled.token_as_str().unwrap(), "s");
}

#[tokio::test]
async fn show_top_logprobs_chat() {
    // Given
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let message = Message::user("An apple a day");

    // When
    let task = TaskChat {
        messages: vec![message],
        stopping: Stopping::from_maximum_tokens(1),
        sampling: ChatSampling::MOST_LIKELY,
        logprobs: Logprobs::Top(2),
    };

    let response = client.chat(&task, model, &How::default()).await.unwrap();

    // Then
    assert_eq!(response.logprobs.len(), 1);
    assert_eq!(
        response.logprobs[0].sampled.token_as_str().unwrap(),
        " Keep"
    );
    assert_eq!(response.logprobs[0].top.len(), 2);
    assert_eq!(response.logprobs[0].top[0].token_as_str().unwrap(), " Keep");
    assert_eq!(
        response.logprobs[0].top[1].token_as_str().unwrap(),
        " keeps"
    );
}

#[tokio::test]
async fn show_logprobs_sampled_completion() {
    // Given
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let task = TaskCompletion::from_text("An apple a day")
        .with_maximum_tokens(2)
        .with_logprobs(Logprobs::Sampled);

    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // // Then
    assert_eq!(response.logprobs.len(), 2);
    assert_eq!(
        response.logprobs[0].sampled.token_as_str().unwrap(),
        " keeps"
    );
    assert!(response.logprobs[0].sampled.logprob.is_sign_negative());
    assert_eq!(response.logprobs[1].sampled.token_as_str().unwrap(), " the");
    assert!(response.logprobs[1].sampled.logprob.is_sign_negative());
}

#[tokio::test]
async fn show_top_logprobs_completion() {
    // Given
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();

    // When
    let task = TaskCompletion::from_text("An apple a day")
        .with_maximum_tokens(1)
        .with_logprobs(Logprobs::Top(2));

    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    assert_eq!(response.logprobs.len(), 1);
    assert_eq!(
        response.logprobs[0].sampled.token_as_str().unwrap(),
        " keeps"
    );
    assert!(response.logprobs[0].sampled.logprob.is_sign_negative());
    assert_eq!(response.logprobs[0].top.len(), 2);
    assert_eq!(
        response.logprobs[0].top[0].token_as_str().unwrap(),
        " keeps"
    );
    assert_eq!(response.logprobs[0].top[1].token_as_str().unwrap(), " may");
    assert!(response.logprobs[0].top[0].logprob > response.logprobs[0].top[1].logprob);
}

#[tokio::test]
async fn show_token_usage_chat() {
    // Given
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let message = Message::user("An apple a day");

    let task = TaskChat {
        messages: vec![message],
        stopping: Stopping::from_maximum_tokens(3),
        sampling: ChatSampling::MOST_LIKELY,
        logprobs: Logprobs::No,
    };

    // When
    let response = client.chat(&task, model, &How::default()).await.unwrap();

    // Then
    assert_eq!(response.usage.prompt_tokens, 19);
    assert_eq!(response.usage.completion_tokens, 3);
}

#[tokio::test]
async fn show_token_usage_completion() {
    // Given
    let model = "pharia-1-llm-7b-control";
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let task = TaskCompletion::from_text("An apple a day")
        .with_maximum_tokens(3)
        .with_logprobs(Logprobs::No);

    // When
    let response = client
        .completion(&task, model, &How::default())
        .await
        .unwrap();

    // Then
    assert_eq!(response.usage.prompt_tokens, 5);
    assert_eq!(response.usage.completion_tokens, 3);
}

#[tokio::test]
async fn trace_context_is_propagated() {
    // Given a client with a trace context
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
    let span_id = 0x00f067aa0ba902b7;
    let trace_context = TraceContext::new_sampled(trace_id, span_id, None);

    // When the client is used to make a request
    let task = TaskCompletion::from_text("Hello, World!");

    // Then the completion succeeds
    let response = client
        .completion(
            &task,
            "pharia-1-llm-7b-control",
            &How {
                trace_context: Some(trace_context),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Then the response is non-empty
    assert!(!response.completion.is_empty());
}

#[tokio::test]
async fn unknown_model_gives_model_not_found_error() {
    // Given
    let client = Client::with_auth(inference_url(), pharia_ai_token()).unwrap();
    let task = TaskCompletion::from_text("Hello, World!");

    // When
    let result = client
        .completion(&task, "pharia-9", &How::default())
        .await
        .unwrap_err();

    // Then we get a model not found error
    assert!(matches!(result, Error::ModelNotFound));
}
