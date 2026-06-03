use mangaba::core::tools::*;
use serde_json::json;

#[tokio::test]
async fn test_calculator_tool() {
    let tool = CalculatorTool;
    let result = tool.call(json!({"expression": "2 + 3 * 4"})).await.unwrap();
    assert!(result.success);
    assert_eq!(result.output.unwrap()["result"], json!(14.0));
}

#[tokio::test]
async fn test_calculator_division_by_zero() {
    let tool = CalculatorTool;
    let result = tool.call(json!({"expression": "1 / 0"})).await.unwrap();
    assert!(!result.success);
}

#[tokio::test]
async fn test_word_counter() {
    let tool = WordCounterTool;
    let result = tool.call(json!({"text": "Hello world. How are you?"})).await.unwrap();
    assert!(result.success);
    let output = result.output.unwrap();
    assert_eq!(output["words"], 5);
    assert_eq!(output["sentences"], 2);
}

#[tokio::test]
async fn test_echo_tool() {
    let tool = EchoTool;
    let result = tool.call(json!({"hello": "world"})).await.unwrap();
    assert!(result.success);
    assert_eq!(result.output.unwrap()["hello"], "world");
}

#[tokio::test]
async fn test_text_splitter() {
    let tool = TextSplitterTool;
    let text = "A. B. C. D. E. F. G. H. I. J. K.";
    let result = tool.call(json!({"text": text, "chunk_size": 10, "chunk_overlap": 0})).await.unwrap();
    assert!(result.success);
    let output = result.output.unwrap();
    let chunks = output["chunks"].as_array().unwrap();
    assert!(chunks.len() > 1, "Should produce multiple chunks");
}

#[tokio::test]
async fn test_file_reader_not_found() {
    let tool = FileReaderTool;
    let result = tool.call(json!({"file_path": "/tmp/nonexistent_file_12345.txt"})).await.unwrap();
    assert!(!result.success);
}

#[tokio::test]
async fn test_directory_list() {
    let tool = DirectoryListTool;
    // Use the platform temp dir so the test works on Windows (no `/tmp`) too.
    let dir = std::env::temp_dir();
    let result = tool
        .call(json!({"directory_path": dir.to_str().unwrap()}))
        .await
        .unwrap();
    assert!(result.success);
}
