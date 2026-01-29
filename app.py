# -*- coding: utf-8 -*-
"""Hermes-3-Llama-3.1-8B WebUI - 文本生成与对话可视化界面"""
import gradio as gr
import json
import time
from typing import Tuple

MODEL_NAME = "Hermes-3-Llama-3.1-8B"

def generate_text(
    user_input: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    mode: str
) -> Tuple[str, str]:
    """生成文本回复（演示模式，不加载真实模型）"""
    if not user_input.strip():
        return "请输入您的消息。", json.dumps({
            "status": "等待输入",
            "mode": "demo",
            "tokens": 0
        }, ensure_ascii=False, indent=2)
    
    # 演示输出
    demo_response = f"""这是 {MODEL_NAME} 的演示输出。

**用户输入：** {user_input}

**系统提示：** {system_prompt if system_prompt else "默认系统提示"}

**生成模式：** {mode}
**参数设置：** temperature={temperature:.2f}, top_p={top_p:.2f}, max_tokens={max_tokens}

**模型回复（演示）：**
根据您的输入，模型将生成相应的回复。在实际部署中，这里会显示模型的实际生成结果。

Hermes-3-Llama-3.1-8B 是一个强大的对话模型，支持：
- 多轮对话
- 函数调用
- JSON 模式输出
- 角色扮演
- 长文本生成

当前为演示模式，未加载真实模型权重。加载模型后，将在此处显示实际的生成结果。"""

    metrics = {
        "status": "demo_mode",
        "mode": mode,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "estimated_tokens": min(max_tokens, 512),
        "note": "演示模式：未加载真实模型"
    }
    
    return demo_response, json.dumps(metrics, ensure_ascii=False, indent=2)

def load_model_status():
    """返回模型加载状态"""
    return "**模型状态：** 演示模式\n\n当前界面为前端展示，未加载真实模型权重。在实际部署中，点击「加载模型」按钮将下载并加载 Hermes-3-Llama-3.1-8B 模型。"

with gr.Blocks(
    title=f"{MODEL_NAME} WebUI",
    theme=gr.themes.Soft(),
    css="""
    .model-info { padding: 15px; background: #f0f0f0; border-radius: 5px; margin: 10px 0; }
    """
) as demo:
    gr.Markdown(f"""
    # {MODEL_NAME} WebUI
    
    基于 Llama-3.1-8B 架构的 Hermes 3 系列对话模型可视化界面
    
    **模型特性：**
    - 支持 ChatML 格式的多轮对话
    - 函数调用（Function Calling）能力
    - JSON 模式结构化输出
    - 角色扮演与长文本生成
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            model_status = gr.Markdown(load_model_status(), elem_classes="model-info")
            load_btn = gr.Button("加载模型（演示）", variant="secondary")
            
            gr.Markdown("### 对话设置")
            system_prompt = gr.Textbox(
                label="系统提示（System Prompt）",
                placeholder="例如：你是一个有用的AI助手...",
                value="你是一个有用的AI助手，能够理解和回答各种问题。",
                lines=3
            )
            
            mode = gr.Radio(
                choices=["对话模式", "函数调用模式", "JSON模式"],
                value="对话模式",
                label="生成模式"
            )
            
            gr.Markdown("### 生成参数")
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature（温度）"
            )
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p（核采样）"
            )
            max_tokens = gr.Slider(
                minimum=64,
                maximum=2048,
                value=512,
                step=64,
                label="最大生成长度"
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 对话界面")
            user_input = gr.Textbox(
                label="输入消息",
                placeholder="请输入您的消息...",
                lines=5
            )
            generate_btn = gr.Button("生成回复", variant="primary", size="lg")
            
            output_text = gr.Textbox(
                label="模型回复",
                lines=15,
                interactive=False
            )
            
            metrics_output = gr.Code(
                label="生成指标",
                language="json",
                lines=10
            )
    
    # 事件绑定
    generate_btn.click(
        fn=generate_text,
        inputs=[user_input, system_prompt, temperature, top_p, max_tokens, mode],
        outputs=[output_text, metrics_output]
    )
    
    load_btn.click(
        fn=lambda: "**模型状态：** 演示模式\n\n在实际部署中，点击此按钮将下载并加载模型。当前为演示界面，不执行真实模型加载。",
        outputs=[model_status]
    )
    
    gr.Markdown("""
    ---
    **使用说明：**
    - 本界面为演示模式，展示 Hermes-3-Llama-3.1-8B 的交互流程
    - 在实际部署中，加载模型后将执行真实的文本生成
    - 支持 ChatML 格式，可通过系统提示控制模型行为
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
