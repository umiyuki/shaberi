import sys
from jinja2 import Template, FileSystemLoader, Environment

def render_template(template_file, messages):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_file)
    return template.render(messages=messages, add_generation_prompt=True)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <template_file>")
        sys.exit(1)

    template_file = sys.argv[1]
    messages = [{"role": "user", "content": "まどか☆マギカで一番可愛いキャラは？"}]

    try:
        prompt = render_template(template_file, messages)
        print(prompt)
    except Exception as e:
        print(f"Error: {str(e)}")