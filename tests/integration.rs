use std::{fs::File, io::BufReader, sync::OnceLock};

use aleph_alpha_client::{
    cosine_similarity, Client, Granularity, How, ImageScore, ItemExplanation, Modality, Prompt,
    PromptGranularity, Role, Sampling, SemanticRepresentation, Stopping, Task,
    TaskBatchSemanticEmbedding, TaskChat, TaskCompletion, TaskDetokenization, TaskExplanation,
    TaskSemanticEmbedding, TaskTokenization, TextScore,
};
use dotenv::dotenv;
use image::ImageFormat;

fn api_token() -> &'static str {
    static AA_API_TOKEN: OnceLock<String> = OnceLock::new();
    AA_API_TOKEN.get_or_init(|| {
        drop(dotenv());
        std::env::var("AA_API_TOKEN")
            .expect("AA_API_TOKEN environment variable must be specified to run tests.")
    })
}

#[tokio::test]
async fn chat_with_pharia_1_7b_base() {
    // When
    let task = TaskChat::new(Role::System, "Instructions").append_message(Role::User, "Question");

    let model = "pharia-1-llm-7b-control";
    let client = Client::with_authentication(api_token()).unwrap();
    let response = client.chat(&task, model, &How::default()).await.unwrap();

    // Then
    assert!(!response.message.content.is_empty())
}

#[tokio::test]
async fn completion_with_luminous_base() {
    // When
    let task = TaskCompletion::from_text("Hello").with_maximum_tokens(1);

    let model = "luminous-base";
    let client = Client::with_authentication(api_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    eprintln!("{}", response.completion);

    // Then
    assert!(!response.completion.is_empty())
}

#[tokio::test]
async fn request_authentication_has_priority() {
    let bad_aa_api_token = "DUMMY";
    let task = TaskCompletion::from_text("Hello").with_maximum_tokens(1);

    let model = "luminous-base";
    let client = Client::with_authentication(bad_aa_api_token).unwrap();
    let response = client
        .output_of(
            &task.with_model(model),
            &How {
                api_token: Some(api_token().to_owned()),
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
    let client = Client::new("https://api.aleph-alpha.com".to_owned(), None).unwrap();
    let response = client
        .output_of(
            &task.with_model(model),
            &How {
                api_token: Some(api_token().to_owned()),
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
    let client = Client::new("https://api.aleph-alpha.com".to_owned(), None).unwrap();
    client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then the client panics on invocation
}

#[tokio::test]
async fn semanitc_search_with_luminous_base() {
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
    let client = Client::with_authentication(api_token()).unwrap();

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
    };
    let model = "luminous-base";
    let client = Client::with_authentication(api_token()).unwrap();
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
    };
    let model = "luminous-base";
    let client = Client::with_authentication(api_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    assert!(!response.completion.is_empty());
    assert_eq!(response.completion, " I am doing fine, how are you?\n");
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
    let client = Client::with_authentication(api_token()).unwrap();

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
    let client = Client::with_authentication(api_token()).unwrap();

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
    let client = Client::with_authentication(api_token()).unwrap();

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
    };
    let model = "luminous-base";
    let client = Client::with_authentication(api_token()).unwrap();
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
    };
    let model = "luminous-base";
    let client = Client::with_authentication(api_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    eprintln!("{}", response.completion);
    assert!(response.completion.contains("cat"))
}

#[tokio::test]
async fn only_answer_with_specific_animal() {
    // Given
    let prompt = Prompt::from_text("What is your favorite animal?");

    // When
    let task = TaskCompletion {
        prompt,
        stopping: Stopping::from_maximum_tokens(1),
        sampling: Sampling {
            start_with_one_of: &[" dog"],
            ..Default::default()
        },
    };
    let model = "luminous-base";
    let client = Client::with_authentication(api_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    eprintln!("{}", response.completion);
    assert_eq!(response.completion, " dog");
}

#[should_panic]
#[tokio::test]
async fn answer_should_continue() {
    // Given
    let prompt = Prompt::from_text("Knock knock. Who's there?");

    // When
    let task = TaskCompletion {
        prompt,
        stopping: Stopping::from_maximum_tokens(20),
        sampling: Sampling {
            start_with_one_of: &[" Says.", " Art.", " Weekend."],
            ..Default::default()
        },
    };
    let model = "luminous-base";
    let client = Client::with_authentication(api_token()).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    eprintln!("{}", response.completion);
    assert!(response.completion.starts_with(" Says."));
    assert!(response.completion.len() > " Says.".len());
}

#[tokio::test]
async fn batch_semanitc_embed_with_luminous_base() {
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

    let client = Client::with_authentication(api_token()).unwrap();

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
async fn tokenization_with_luminous_base() {
    // Given
    let input = "Hello, World!";

    let client = Client::with_authentication(api_token()).unwrap();

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

    let client = Client::with_authentication(api_token()).unwrap();

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
    let client = Client::with_authentication(api_token()).unwrap();

    // When
    let tokenizer = client
        .tokenizer_by_model("Pharia-1-LLM-7B-control", None)
        .await
        .unwrap();

    // Then
    assert_eq!(128_000, tokenizer.get_vocab_size(true));
}
