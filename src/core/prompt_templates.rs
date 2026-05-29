//! Prompt template engine and system prompt builder.
//!
//! [`PromptTemplate`] parses `{variable}` placeholders and renders with a value map.
//! [`SystemPromptBuilder`] constructs agent system prompts from role, goal, backstory,
//! tool descriptions, and optional sections.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PromptTemplate {
    pub template: String,
    pub variables: Vec<String>,
}

impl PromptTemplate {
    pub fn new(template: &str) -> Self {
        let mut variables = Vec::new();
        let mut in_var = false;
        let mut var_name = String::new();
        for ch in template.chars() {
            match ch {
                '{' => in_var = true,
                '}' if in_var => {
                    variables.push(var_name.clone());
                    var_name.clear();
                    in_var = false;
                }
                _ if in_var => var_name.push(ch),
                _ => {}
            }
        }
        Self { template: template.to_string(), variables }
    }

    pub fn render(&self, values: &HashMap<&str, &str>) -> String {
        let mut result = self.template.clone();
        for (key, val) in values {
            result = result.replace(&format!("{{{}}}", key), val);
        }
        result
    }
}

pub struct SystemPromptBuilder {
    parts: Vec<String>,
}

impl SystemPromptBuilder {
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    pub fn role(mut self, role: &str) -> Self {
        self.parts.push(format!("You are: {}", role));
        self
    }

    pub fn goal(mut self, goal: &str) -> Self {
        self.parts.push(format!("Your goal is: {}", goal));
        self
    }

    pub fn backstory(mut self, backstory: &str) -> Self {
        self.parts.push(format!("Background: {}", backstory));
        self
    }

    pub fn tools(mut self, tool_descriptions: &[String]) -> Self {
        if !tool_descriptions.is_empty() {
            self.parts.push(format!("Available tools:\n{}", tool_descriptions.join("\n")));
        }
        self
    }

    pub fn section(mut self, title: &str, content: &str) -> Self {
        self.parts.push(format!("{}:\n{}", title, content));
        self
    }

    pub fn build(&self) -> String {
        self.parts.join("\n\n")
    }
}
