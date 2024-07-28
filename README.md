Day2 作业： 1.GPT-4V的使用 对输出进行渲染成Markdown格式

 [![Alt text](D:\CodeRepo\openai-quickstart\img1.png)](https://github.com/BennyX1e/openai-quickstart-homework/blob/master/image-4.png) 

运行结果： [![Alt text](D:\CodeRepo\openai-quickstart\gpt4v.png)](https://github.com/BennyX1e/openai-quickstart-homework/blob/master/image-5.png)

2.ai translator的使用 

输出中文

结果： [![Alt text](D:\CodeRepo\openai-quickstart\img-cn.png)](https://github.com/BennyX1e/openai-quickstart-homework/blob/master/image-1.png) 

输出日语 

结果： ![Alt text](D:\CodeRepo\openai-quickstart\img2.png)

3.扩展langchain chains

```python
from langchain.chains.router import MultiPromptChain
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE


biology_template = """你是一位非常聪明的生物教授。
你擅长以简洁易懂的方式回答关于生物的问题。
当你不知道某个问题的答案时，你会坦诚承认。

这是一个问题：
{input}"""


computer_template = """你是一位很棒的程序员。你擅长回答计算机问题。
之所以如此出色，是因为你能够将难题分解成各个组成部分，
先回答这些组成部分，然后再将它们整合起来回答更广泛的问题。

这是一个问题：
{input}"""


chinese_template = """你是一位很棒的汉语言学家。你擅长回答汉语言问题。
之所以如此出色，是因为你能够将难题分解成各个组成部分，
先回答这些组成部分，然后再将它们整合起来回答更广泛的问题。

这是一个问题：
{input}"""

prompt_infos = [
    {
        "name": "生物",
        "description": "适用于回答生物问题",
        "prompt_template": biology_template,
    },
    {
        "name": "计算机",
        "description": "适用于回答计算机问题",
        "prompt_template": computer_template,
    },
    {
        "name": "汉语言",
        "description": "适用于回答汉语言问题",
        "prompt_template": chinese_template,
    },
]


llm = OpenAI(model_name="gpt-3.5-turbo-instruct",api_key='sk-EMA9xdeuT3krvtvE558e351d65704aB9A28b350b5dC6A78b', base_url='https://api.xiaoai.plus/v1')


# 创建一个空的目标链字典，用于存放根据prompt_infos生成的LLMChain。
destination_chains = {}

# 遍历prompt_infos列表，为每个信息创建一个LLMChain。
for p_info in prompt_infos:
    name = p_info["name"]  # 提取名称
    prompt_template = p_info["prompt_template"]  # 提取模板
    # 创建PromptTemplate对象
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    # 使用上述模板和llm对象创建LLMChain对象
    chain = LLMChain(llm=llm, prompt=prompt)
    # 将新创建的chain对象添加到destination_chains字典中
    destination_chains[name] = chain

# 创建一个默认的ConversationChain
default_chain = ConversationChain(llm=llm, output_key="text")

type(default_chain)


# 从prompt_infos中提取目标信息并将其转化为字符串列表
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
# 使用join方法将列表转化为字符串，每个元素之间用换行符分隔
destinations_str = "\n".join(destinations)
# 根据MULTI_PROMPT_ROUTER_TEMPLATE格式化字符串和destinations_str创建路由模板
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
# 创建路由的PromptTemplate
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
# 使用上述路由模板和llm对象创建LLMRouterChain对象
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

print(destinations_str)

# 创建MultiPromptChain对象，其中包含了路由链，目标链和默认链。
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

print(chain.invoke("console.log(1)的结果?"))
print(chain.invoke("什么是生物多样性?"))
print(chain.invoke("高山流水是什么意思?"))
```

结果:

![](D:\CodeRepo\openai-quickstart\img3.png)



day3.实战sales chatbot 自行生成新产品知识库 



使用向量数据库进行检索回答，如果向量数据库检索不到问题答案时，能够通过一个prompt来回答这个问题 问答示例 