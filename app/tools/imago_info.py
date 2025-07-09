from langchain.tools import tool

with open('tools/data/imago_info.md', 'r', encoding='utf-8') as f:
    IMAGO_INFORMATION = f.read()

@tool
def get_imago_info():
    """Use this tool to retrieve full internal reference information about the Imago ecosystem including apps, websites, the founder, and routes."""
    print("\n[TOOL CALLED 🔧] get_imago_info")
    return IMAGO_INFORMATION