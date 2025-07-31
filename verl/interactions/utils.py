import re

def extract_tag(response, tag):                    
    if f"<{tag}>" in response and f"</{tag}>" in response:
        pattern = f"<{tag}>(.*?)</{tag}>"
        contents = re.findall(pattern, response, re.DOTALL)
        contents = contents[0].strip()
        return contents
    else:
        return None

def process_msg_content(msg_str, tag_list=['think','answer']):
    msg_str = msg_str.lower()
    valid = True
    correct_format = ""
    all_contents = []
    for tag in tag_list:
        start_tag = '<' + tag + '>'
        end_tag = '</' + tag + '>'
        correct_format += start_tag + " ... " + end_tag 
        if not start_tag in msg_str:
            valid = False
        elif not end_tag in msg_str.split(start_tag)[1]:
            valid = False
        else:
            all_contents.append(extract_tag(msg_str, tag))
    return all_contents,valid, 'Could not parse response. Please ensure your response is in the format: ' + correct_format + '.'
    
