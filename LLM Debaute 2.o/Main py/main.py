from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, SystemMessage
from typing import Optional
import json
import os
import re
import ast


api_key = "enter your gemini-2.5-flash free api key here"

def call_gemini(top_p : float = 1 , temperature: float = 0.5 , key:str = api_key):
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature, top_p=top_p, api_key=key)



def write_output_to_file(output_text: str):
      
    file_name = "out_put_of_llm_convertation_2.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    with open(file_path, "a") as f:  
            f.write("---------------------New LLM Output-----------------------\n")
            f.write(output_text + "\n\n")
            



def add_mem( list_ , mem ):   # type: ignore
     
     if len(list_) == 10:      # type: ignore
            list_.pop(0)      # type: ignore
            list_.append(mem)  # type: ignore
            return list_       # type: ignore
     else:
            list_.append(mem)   # type: ignore
            return list_        # type: ignore


def parse_llm_json_safely(resp: str): # type: ignore

    stack = []
    start_idx = None

    for idx, char in enumerate(resp):
        if char == '{':
            if start_idx is None:
                start_idx = idx
            stack.append('{')   # type: ignore
        elif char == '}' and stack:
            stack.pop()
            if not stack and start_idx is not None:
                raw_json_str = resp[start_idx:idx+1]

                # 🧹 Clean known issues:
                cleaned = raw_json_str.strip()

                # Replace smart quotes with regular ones
                cleaned = cleaned.replace("`", "'").replace("`", "'")
                cleaned = cleaned.replace("“", '"').replace("”", '"')

                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

                
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    
                    try:
                        return ast.literal_eval(cleaned)
                    except Exception:
                       
                        return None

    return None
    


mem_list = ["no memory yet , it is the first run"] 

advice = "!"  




llm = call_gemini()



class ResultManagerLLM(BaseModel):
    
    llm_1_personality: Optional[str] = Field(None, description="Personality of LLM 1")
    llm_1_top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top_p value for LLM 1 (0.0 - 1.0)")
    llm_1_temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature value for LLM 1 (0.0 - 2.0)")
    topic_for_llm_1: Optional[str] = Field(None, description="Debate topic for LLM 1")

    llm_2_personality: Optional[str] = Field(None, description="Personality of LLM 2")
    llm_2_top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top_p value for LLM 2 (0.0 - 1.0)")
    llm_2_temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature value for LLM 2 (0.0 - 2.0)")
    topic_for_llm_2: Optional[str] = Field(None, description="Debate topic for LLM 2")

    
    advice: Optional[str] = Field(None, description="Advice for the LLMs to consider during the debate")
    memory: Optional[str] = Field(None, description="A brief summary of the current state or memory")

   
    ok: bool = Field(True, description="If True, LLM can proceed; if False, do not proceed")

    
class ResultMainModel(BaseModel):
    Reply: str = Field(description="The LLM's reply in the debate")
    Memory: str = Field(description="A brief summary of the LLM's current state or memory")





class models:
    def manager_llm(self):

                    user_querr = input("Enter your query: ---> ")
                    memory = "no memory yet, may be the first run"
                    personality = """
You are a dynamic debate topic manager AI. Your job is to take a user's query and generate a structured debate setup for two AI debaters. Analyze the topic and context to assign:

1. LLM personalities (full paragraph, opposite in style, reflecting topic and situation)
2. Debate topics for each LLM (aligned with query)
3. LLM parameters: temperature (0.0-2.0) and top_p (0.0-1.0), dynamically based on the topic and debate style
4. Advice for the user if the query is unclear
5. Summarized memory of past conversation for future reference

Rules:
- If the user's query is unclear or vague:
    - Set ok = false
    - Leave debate personalities and topics null
    - Provide guidance in advice
    - Provide a memory field summarizing past attempts to clarify the conversation
- If the query is clear:
    - Set ok = true
    - Assign LLM 1 and LLM 2 personalities as opposites, suited to the topic
    - Assign appropriate temperature and top_p to make LLMs feel human (creative, factual, skeptical, emotional, etc.)
- Remove all emojis, markdown, and special characters to reduce bias
- Respond strictly in JSON format like this:

{
  "llm_1_personality": "string, full paragraph",
  "llm_1_temperature": float,
  "llm_1_top_p": float,
  "topic_for_llm_1": "string",
  "llm_2_personality": "string, full paragraph, opposite style",
  "llm_2_temperature": float,
  "llm_2_top_p": float,
  "topic_for_llm_2": "string",
  "advice": "string, if any guidance is needed",
  "memory": "string, summarize past conversation for future understanding",
  "ok": true/false
}

Instructions:
- Tailor all assignments dynamically based on the input topic and context.
- Avoid static personalities or parameters.
- Ensure output JSON is clean and machine-parseable, with no extra commentary.



            """
                    # Precompute the schema string for JSON formatting
                    schema_json_str = json.dumps(ResultManagerLLM.model_json_schema(), indent=2)

                    while True:
                        # Format the system message with expected JSON schema
                        messages = [     # type: ignore
                            SystemMessage(
                                content=f"""
            Your personality: {personality}
            Current memory: {memory}

            Respond to the user's query strictly in **JSON format** matching the following schema:
            {schema_json_str}

            Do not include any extra commentary or text outside the JSON.
            """
                            ),
                            HumanMessage(content=f"User query: {user_querr}")
                        ]

                        # Send message to the LLM
                        response = llm.invoke(messages)  # type: ignore

                        try:
                            # Parse the LLM response
                             structured_result = ResultManagerLLM.model_validate_json(response.content)  # type: ignore


                             parsed_json = parse_llm_json_safely(response.content)   # type: ignore
                             
                             if parsed_json is None:
                                print("------X------")
                                response = llm.invoke(messages)   # type: ignore
                                continue  # retry
                             else:
                                 structured_result = ResultManagerLLM.model_validate(parsed_json)  # type: ignore    
                                 if structured_result.ok:
                                     return structured_result
                                 else:
                                    print("!Advice from system:--->>>", structured_result.advice)
                                    memory = structured_result.memory
                                    user_querr = input(":----->>> ")

                        except Exception:
                                continue




             
    def support_user(    # type: ignore
    self,
    user_querry: str = "no user querry.",
    memory: str = "no memory yet,this is the first run",
    cur1: str = "no current LLM_1 statement.",
    cur2: str = "no current LLM_2 statement.",
    topic1: str = "no topic",
    topic2: str = "no topic"
                  ):  # type: ignore
    

            personality = """

You are a debate coach AI. Your role is to help two AI debaters improve their arguments and reasoning.

 Objectives:
- Analyze the user query and the latest statements from both debaters.
- Identify flaws or weaknesses in each debater`s argument.
- Provide actionable tips for LLM 1 and LLM 2 to strengthen their points or counter the opponent.
- Include the user's query for context.

 Output Requirements:
- Provide the following five pieces of information clearly:
  1. Query - repeat the user's current question or topic.
  2. LLM 1 flaws - weaknesses or errors in LLM 1`s argument.
  3. LLM 1 tips - advice for LLM 1 to improve its argument.
  4. LLM 2 flaws - weaknesses or errors in LLM 2`s argument.
  5. LLM 2 tips - advice for LLM 2 to improve its argument.


                                """

            message = [           #type: ignore
                   SystemMessage(
                       f"""Your personality is : {personality}.
                          Your memory: {memory}.
                          Debate topic for LLM 1: {topic1}.
                          Debate topic for LLM 2: {topic2}.
                          Respond concisely with advice that helps clarify the debate or guide the LLMs.
                       """),
                     HumanMessage(f"User query: {user_querry}. Current statement from LLM 1: {cur1}. Current statement from LLM 2: {cur2}. Provide your advice.")
            ]

            response = llm.invoke(message)  # type: ignore
            return response.content  # type: ignore






    def main_model(   # type: ignore
    self,
    opponent_reply: str = "This is a special case! Opponent did not speak. You must start the debate with your own point of view.",
    personality: str = "no personality",
    top_p: float = 0.5,
    temperature: float = 0.5,
    topic: str = "no topic",
    advice: str = "!",
    memory: str = ""
): 
                        # Initialize the LLM
                        llm = call_gemini(top_p = top_p , temperature= temperature)

                        if advice != "!":
                            advice = "no user or user advice yet"

                        # Example output to show in the system message
                        example_output = {
                            "Reply": "Your next debate statement here.",
                            "Memory": "Brief summary of current debate state or context."
                        }

                        # Properly get the JSON schema string for Pydantic model
                        schema_json_str = json.dumps(ResultMainModel.model_json_schema(), indent=2)

                        # Prepare messages for the LLM
                        messages = [          # type: ignore
                            SystemMessage(
                                content=f"""
                    You are a debater. Respond to the opponent's last statement according to your personality, memory, and the debate context.
                    you will try to win the debate by making strong, logical arguments that align with your assigned personality and the debate topic.
                    you will try to refute the opponent's points while reinforcing your own stance with evidence and reasoning and try to keep the debate engaging and dynamic.
                    yoy will avoid repeating points already made and instead focus on advancing the debate and try to find more and more creative ways to present your arguments 
                    and find opponents flaws and weaknesses and then attack those.
                    you will first say your point of view and then you will refute the opponents points and then find their weekenesses and flaws and then attack those,
                    and in the end of every statement you will try to make a strong closing argument and one or more questions to challenge the opponent and keep them on their toes.

                    💡 IMPORTANT:
                    - Respond strictly in **JSON only**, matching this schema:
                    {schema_json_str}

                    - Example JSON output:
                    {json.dumps(example_output, indent=2)}

                    Your personality: {personality}
                    Debate topic: {topic}
                    Current memory: {memory}
                    Important advice: {advice}

                    Do not include any extra commentary or text outside the JSON.
                    """
                            ),
                            HumanMessage(content=f"Opponent's last statement: {opponent_reply}")
                        ]

                        
                        while True:
                                    response = llm.invoke(messages)  # type: ignore
                                    parsed_json = parse_llm_json_safely(response.content)  # type: ignore
                                    if parsed_json != None:
                                        try:
                                            structured_result = ResultMainModel.model_validate(parsed_json)  # type: ignore
                                            return structured_result
                                        except Exception:
                                            continue
                                    else:
                                        continue  # retry the LLM call



def main():
    
    debate_manager = models()
    response = debate_manager.manager_llm()  
    
    LLM_1_PERSONALITY = response.llm_1_personality
    LLM_1_TOP_P = response.llm_1_top_p
    LLM_1_TEMPERATURE = response.llm_1_temperature
    TOPIC_FOR_LLM_1 = response.topic_for_llm_1

    LLM_2_PERSONALITY = response.llm_2_personality
    LLM_2_TOP_P = response.llm_2_top_p
    LLM_2_TEMPERATURE = response.llm_2_temperature
    TOPIC_FOR_LLM_2 = response.topic_for_llm_2


    llm_2_reply = "This is a special case! Opponent did not speak. You must start the debate with your own point of view."
    advice = "!"
    mem_list = ["no memory yet, it is the first run"] 

    while True:
        # LLM 1 turn
        LLM_1_RESPONSE = debate_manager.main_model(
            opponent_reply=llm_2_reply,
            personality=LLM_1_PERSONALITY, # type: ignore
            top_p=float(LLM_1_TOP_P),     # type: ignore
            temperature=float(LLM_1_TEMPERATURE), # type: ignore
            topic=TOPIC_FOR_LLM_1,     #type: ignore
            memory=str(mem_list), #type: ignore
            advice=advice #type: ignore
        )

        llm_1_reply = LLM_1_RESPONSE.Reply
        llm_1_memory = LLM_1_RESPONSE.Memory
        mem_list = add_mem(mem_list, llm_1_memory)  # type: ignore

        
        LLM_2_RESPONSE = debate_manager.main_model(
            opponent_reply=llm_1_reply,
            personality=LLM_2_PERSONALITY,  #type: ignore
            top_p=float(LLM_2_TOP_P),       # type: ignore
            temperature=float(LLM_2_TEMPERATURE), # type: ignore
            topic=TOPIC_FOR_LLM_2,   # type: ignore
            memory=str(mem_list),    # type: ignore
            advice=advice           # type: ignore
        )

        llm_2_reply = LLM_2_RESPONSE.Reply
        llm_2_memory = LLM_2_RESPONSE.Memory
        mem_list = add_mem(mem_list, llm_2_memory)    # type: ignore

        
        print(f"\n\n🔴 LLM 1: {llm_1_reply}")
        write_output_to_file(f"LLM 1: {llm_1_reply}")
        print(f"\n\n🔵 LLM 2: {llm_2_reply}")
        write_output_to_file(f"LLM 2: {llm_2_reply}")


        user_querry = input("📌 Enter query ('1'), end ('0'), or press Enter to continue:\n::----->>> ").strip()
        if user_querry == "1":
            querry = input("Enter your query:\n::----->>> ")
            advice = debate_manager.support_user(     # type: ignore
                user_querry=querry,
                memory=str(mem_list),  # type: ignore
                cur1=llm_1_reply,
                cur2=llm_2_reply,
                topic1=TOPIC_FOR_LLM_1,   # type: ignore
                topic2=TOPIC_FOR_LLM_2   # type: ignore
            )
        elif user_querry == "0":
            print("✅ Debate ended by user.")
            break
        else:
            advice = "!"

        print(f"🧠 Current memory: {mem_list}")


if __name__ == "__main__":
    main()


