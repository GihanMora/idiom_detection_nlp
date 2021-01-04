import pymongo
from selenium.common.exceptions import TimeoutException, WebDriverException
import requests
import time
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from bs4 import BeautifulSoup
from bs4.element import Tag
from fake_useragent import UserAgent
from random import choice
import pandas as pd
def get_browser():
    ua = UserAgent()
    # PROXY = proxy_generator()
    for k in range(5):
        userAgent = ua.random  # get a random user agent
        if ('Mobile' in userAgent):
            print("got mobile")
            continue
        else:
            break
    print(userAgent)
    options = webdriver.ChromeOptions()  # use headless version of chrome to avoid getting blocked
    options.add_argument('headless')
    options.add_argument(f'user-agent={userAgent}')
    # options.add_argument("start-maximized")# // open Browser in maximized mode
    # options.add_argument("disable-infobars")# // disabling infobars
    # options.add_argument("--disable-extensions")# // disabling extensions
    # options.add_argument("--disable-gpu")# // applicable to windows os only
    # options.add_argument("--disable-dev-shm-usage")# // overcome limited resource problems
    # options.add_argument('--proxy-server=%s' % PROXY)
    browser = webdriver.Chrome(chrome_options=options,  # give the path to selenium executable
                                   # executable_path='F://Armitage_lead_generation_project//chromedriver.exe'
                                executable_path='utilities//chromedriver.exe',
                                    # service_args=["--verbose", "--log-path=D:\\qc1.log"]
                                   )
    return browser


def scrape(comp_url):

    browser = get_browser()
    print('check1')
    # browser.set_page_load_timeout(30)

    browser.get(comp_url)
    time.sleep(5)
    pageSource = browser.page_source
    # print(pageSource)
    time.sleep(5)


    soup = BeautifulSoup(pageSource, 'html.parser')#bs4
    idioms_seg = soup.findAll('dt')
    dd_seg = soup.findAll('dd')
    example_list = []
    idioms_list = []
    meaning_list = []
    idiomic_part_list = []
    erros = []
    for k in idioms_seg:
        idioms_list.append(k.text)
        # p_text = k.text
        # if('Meaning:' in p_text):
        #     meaning = p_text
        # if('Example:' in p_text):
        #     example =
    for i,dd in enumerate(dd_seg):
        p_seg = dd.findAll('p')
        for k_i in p_seg:
            print(k_i)
            p_text = k_i.text
            if (('Meaning:' in p_text) and 'Example:' in p_text):
                print('both in ', p_text)
                erros.append(idioms_list[i])
            elif('Meaning:' in p_text):
                meaning = p_text
                meaning_list.append(meaning)
            elif('Example:' in p_text):
                example = p_text
                idiomic_part = k_i.findAll('strong')[-1].text
                example_list.append(example)
                idiomic_part_list.append(idiomic_part)

            else:
                print('error in this id',i,idioms_list[i])
                erros.append(idioms_list[i])
                # idioms_list.remove()
    erros = list(set(erros))
    for er in erros:
        idioms_list.remove(er)
    browser.quit()

    print(idioms_list)
    print(meaning_list)
    print(example_list)
    print(idiomic_part_list)

    return {'idioms': idioms_list, 'meanings': meaning_list, 'examples': example_list, 'idiomics': idiomic_part_list}


    # print(k.text)
output_df = pd.DataFrame()
e_example_list = []
e_idioms_list = []
e_meaning_list = []
e_idiomic_part_list = []
for i in range(110,142):
    print(i)
    comp_url = 'https://www.theidioms.com/list/page/'+str(i)+'/'
    results = scrape(comp_url)
    print(results)
    if(len(results['examples'])==len(results['idioms'])==len(results['meanings'])==len(results['idiomics'])):
        e_example_list = e_example_list +results['examples']
        e_idioms_list = e_idioms_list + results['idioms']
        e_meaning_list = e_meaning_list + results['meanings']
        e_idiomic_part_list = e_idiomic_part_list + results['idiomics']
        print([len(results['examples']),len(results['idioms']),len(results['meanings']),len(results['idiomics'])])
    else:
        print("*********************************error************************")

output_df['idiom']=e_idioms_list
output_df['meaning'] = e_meaning_list
output_df['example']= e_example_list
output_df['idiomatic_part'] = e_idiomic_part_list
print(output_df.head(10))
output_df.to_csv('Out110_142.csv')

