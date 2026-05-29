use async_trait::async_trait;
use anyhow::{Result, anyhow};
use serde_json::{json, Value};
use crate::core::types::ToolResult;
use super::BaseTool;

fn safe_eval(expr: &str) -> Result<f64> {
    let tokens = tokenize(expr)?;
    let mut parser = Parser::new(tokens);
    let result = parser.parse_expression()?;
    if !parser.is_at_end() {
        return Err(anyhow!("Unexpected token after expression"));
    }
    Ok(result)
}

#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' => i += 1,
            '+' => { tokens.push(Token::Plus); i += 1; }
            '-' => { tokens.push(Token::Minus); i += 1; }
            '*' => { tokens.push(Token::Star); i += 1; }
            '/' => { tokens.push(Token::Slash); i += 1; }
            '%' => { tokens.push(Token::Percent); i += 1; }
            '^' => { tokens.push(Token::Caret); i += 1; }
            '(' => { tokens.push(Token::LParen); i += 1; }
            ')' => { tokens.push(Token::RParen); i += 1; }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let num: f64 = num_str.parse().map_err(|_| anyhow!("Invalid number: {}", num_str))?;
                tokens.push(Token::Number(num));
            }
            c => return Err(anyhow!("Unexpected character: {}", c)),
        }
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        self.pos += 1;
        t
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn parse_expression(&mut self) -> Result<f64> {
        // expr = term (('+' | '-') term)*
        let mut left = self.parse_term()?;
        while let Some(op) = self.peek() {
            match op {
                Token::Plus => { self.advance(); left += self.parse_term()?; }
                Token::Minus => { self.advance(); left -= self.parse_term()?; }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<f64> {
        // term = unary (('*' | '/' | '%') unary)*
        let mut left = self.parse_unary()?;
        while let Some(op) = self.peek() {
            match op {
                Token::Star => { self.advance(); left *= self.parse_unary()?; }
                Token::Slash => {
                    self.advance();
                    let right = self.parse_unary()?;
                    if right == 0.0 { return Err(anyhow!("Division by zero")); }
                    left /= right;
                }
                Token::Percent => {
                    self.advance();
                    let right = self.parse_unary()?;
                    if right == 0.0 { return Err(anyhow!("Division by zero")); }
                    left %= right;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<f64> {
        match self.peek() {
            Some(Token::Minus) => { self.advance(); Ok(-self.parse_power()?) }
            Some(Token::Plus) => { self.advance(); self.parse_power() }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> Result<f64> {
        // power = atom ('^' unary)?
        let mut base = self.parse_atom()?;
        if let Some(Token::Caret) = self.peek() {
            self.advance();
            let exp = self.parse_unary()?;
            base = base.powf(exp);
        }
        Ok(base)
    }

    fn parse_atom(&mut self) -> Result<f64> {
        match self.advance() {
            Some(Token::Number(n)) => Ok(*n),
            Some(Token::LParen) => {
                let val = self.parse_expression()?;
                match self.peek() {
                    Some(Token::RParen) => { self.advance(); Ok(val) }
                    _ => Err(anyhow!("Missing closing parenthesis")),
                }
            }
            Some(t) => Err(anyhow!("Unexpected token: {:?}", t)),
            None => Err(anyhow!("Unexpected end of expression")),
        }
    }
}

pub struct CalculatorTool;

#[async_trait]
impl BaseTool for CalculatorTool {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str { "Evaluate a mathematical expression and return the numeric result. Supports +, -, *, /, %, ^ and parentheses." }
    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. '2 + 3 * 4'"
                }
            },
            "required": ["expression"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let expression = args.get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'expression' argument"))?;

        match safe_eval(expression) {
            Ok(result) => Ok(ToolResult {
                call_id: "calc".to_string(),
                tool_name: "calculator".to_string(),
                output: Some(json!({"result": result})),
                error: None,
                success: true,
            }),
            Err(e) => Ok(ToolResult {
                call_id: "calc".to_string(),
                tool_name: "calculator".to_string(),
                output: Some(json!({"error": format!("{}", e)})),
                error: Some(format!("{}", e)),
                success: false,
            }),
        }
    }
}
