import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.rtbf.be/en-continu"
save = "./assets/production.txt"
product = "./assets/prod.txt"
titre=[]
contenu=[]
lien=[]

def getcontenu(contenu): # recupere le contenu dans les paragraphes de chaque articles
        paragraph=''
        for tag in contenu.find_all("p",class_=False):
                if tag.find("em",class_=False):# recupere les citation
                       paragraph+=tag.text+'\n'
                elif tag.find("span",class_=False): # recupere les citation dans les citation
                        paragraph+=tag.text+'\n'
                elif tag.find("strong",class_=False): # recupere les termes en gras ( svt nom )
                       paragraph+=tag.text+'\n'
                else:
                       paragraph+=tag.text+'\n'

                
        print("article copié!")   
        return paragraph


r = requests.get(url)
content = r.content
soup = BeautifulSoup(content,"html.parser")

for tag in soup.find_all("article"):
        titre.append(tag.find("header").text)
        lien.append(tag.find("a").get('href'))

lien = ["https://www.rtbf.be"+ elem for elem in lien if "/article" in elem] # format les urls correctement
print("liste des liens preparé")

df = pd.DataFrame({"Title":[],"Links":[],"Article":[]}) # initie les colones avec les valeurs recupere sur la page d'acceuil
df.loc[:,"Title"]= titre
df.loc[:,"Links"] = lien

for e in lien:
    r = requests.get(e) #requete sur les different article recupere a l'acceuil
    if r.status_code == 200:
        print('recuperation du contenu suivant...')
    newcontent = BeautifulSoup(r.content,"html.parser")
    contenu.append(getcontenu(newcontent))

df.loc[:,"Article"] = contenu
df.to_csv("./assets/prod.csv",index=False,mode='a',header=False)
print("scraping terminé , veuillez verifié le contenu :)")