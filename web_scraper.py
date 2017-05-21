#import the Beautiful soup functions to parse the data returned from the website
from bs4 import BeautifulSoup
import csv
import urllib2
import os
os.chdir('/media/sander/D1-P1/shared_folder/python/RNN')
import re
import numpy as np
import pandas as pd


# Define several cleaning functions
def clean_paragraph(paragraph):

    result=''
    # Remove period halfway through sentences, e.g. 'ca. 1800'.
    paragraph=re.sub(r"(\.)(\s[a-z,0-9])", r"\2", paragraph)
    paragraph=re.sub(r"([A-Z])(\.)(\s[A-Z,0-9])", r"\1 \3", paragraph)

    for line in paragraph.split('.'):
        line_cleaned=clean_line(line)
        if len(line_cleaned)>3:
            result=result+line_cleaned
        else:
            continue
    return result


def clean_line(line):

    line_cleaned=''
    for word in line.split(' '):
        word_cleaned=clean_word(word)
        if len(word_cleaned)>1:
            line_cleaned=line_cleaned+word_cleaned+' '
    return line_cleaned[:-1]+'.'


verwijderlijst=['.',',',':',';','(',')']


def clean_word(word):

    word_cleaned=''
    if word.find('[')>0:
        word=word[:word.find('[')]
    if word.find('[')==0:
        word=' '
    for letter in word:
        if letter not in verwijderlijst:
            word_cleaned=word_cleaned+letter
        else:
            continue
    return word_cleaned


def crawl(current_link):

    global to_visit, visited

    # Trim to_visit list of it becomes too long
    if len(to_visit)>100:
        np.random.choice(to_visit,100)

    # Append current link to visited
    visited.append(current_link)

    # Remove visited links from to_visit
    to_visit=[x for x in to_visit if x not in visited]

    #Query the website and return the html to the variable 'page'
    page = urllib2.urlopen(current_link)

    #Parse the html in the 'page' variable, and store it in Beautiful Soup format
    soup = BeautifulSoup(page, 'html.parser')

    # get all text
    l = soup.find_all('p')

    for index,value in enumerate(l):
        paragraph=l[index].text.encode('utf-8')
        paragraph=clean_paragraph(paragraph)
        for line in paragraph.split('.'):
            if len(line)>3:
                csvWriter.writerow([line+'.'])

    print 'CSV weggeschreven: ' + '\n' + current_link +'\n\n\n'

    # get all links
    all_links=soup.find_all('a')

    # Append useful links to to_visit
    for link in all_links:

            temp = link.get('href')
            try:
                temp=str(temp).encode('utf-8')
            except:
                continue


            if temp.startswith('/wiki/')==True:
                new_link=temp[6:]
                gaan=True
                verwijderen=['Hoofdpagina','Portaal:Navigatie','Wikipedia:Etalage','Categorie:Alles',
                             'Portaal:Gebruikersportaal','Wikipedia:Snelcursus','Portaal:Hulp_en_beheer',
                             'ikipedia','ikimedia','peciaal','.',',',':','jpg','png','bmp','oorverwijs',
                             'source']
                for i in verwijderen:
                    if new_link.find(i)>0:
                        gaan=False
                if gaan==True:
                    new_link='https://nl.wikipedia.org/wiki/'+new_link
                    if new_link not in to_visit and new_link not in visited:
                        to_visit.append(new_link)


# Make list to_visit and visited
to_visit=[]
visited=[]

#specify starting url
current_link = 'https://nl.wikipedia.org/wiki/Ludwig_van_Beethoven'

# write text to CSV
csvFile = open('wiki.csv', 'a')

#Use csv Writer
csvWriter = csv.writer(csvFile)

# Aanroepen crawl functie
crawl(current_link)
while len(visited)<1500:
    # visit next page. pick a random link out of the links
    next_link_index=np.random.random_integers(len(to_visit)-1)
    crawl(to_visit[next_link_index])

test = pd.read_csv('wiki.csv', header=None)