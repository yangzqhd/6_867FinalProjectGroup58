#!/usr/bin/python2.7 -tt

"""
   Parse ESPN Score board data.   

"""

import sys, re, csv
import BeautifulSoup as bs2
import urllib, urllib2, time
from urllib2 import Request, urlopen, URLError, HTTPError
from urlparse import urlparse


def ParsePlyrDataById(league,plyr_id,year):

  Leagues = ['nba','nfl','nhl','mlb']

  # convert the league name to lower
  league = league.lower() 

  if league not in Leagues:
    print "Invalid league...exitting..."
    flag = -1
    return flag
  elif league == 'nba' or league =='nfl':
    flag = ParsePlyrDataByIdNbaNfl(league,plyr_id,year)
  elif league == 'mlb':
    flag = ParsePlyrDataByIdMlb(league,plyr_id,year)

  return flag

def ParsePlyrDataByIdNbaNfl(league,plyr_id,year):

  if plyr_id < 0:
    print "No player info, can't parse data...\n"
    return -1

  request = Request("http://espn.go.com/%s/player/gamelog/_/id/%d/year/%d" %  (league.lower(),plyr_id,year))

  # Try to read the page and handle exceptions if necessary
  # This is especially true for the off-season
  try:
    response = urlopen(request)
  except HTTPError, e:
    print "\tThe server couldn\'t fulfill the request\n"
    print "\tError code:", e.code
    print "\tSorry, there is no data for this player, in this league and year...\n"
    print "\tPlease check the inputs for accuracy...\n"
  except URLError, e:
    print "\tWe failed to reach a server\n"   

  the_page = response.read()
  pool = bs2.BeautifulSoup(the_page)

  title = pool.findAll("title")

  split_title1 = title[0].contents[0].split()
  split_title2 = title[0].contents[0].split('-')

  plyr_name = split_title1[0] + ' ' + split_title1[1]
  team_name = split_title2[1].strip()

  print "Parsing data:\n"
  print "\tPlayer Name\t: %s" % plyr_name
  print "\tLeague\t\t: %s" % league.upper()
  print "\tTeam\t\t: %s" % team_name
  print "\tYear\t\t: %d" % year

  games = pool.findAll("tr")

  NumGames = len(games)

  dummy_games = pool.findAll("tr") # will use this to find postseason start/end

  # figure out which index of dummy_games is 'POSTSEASON' in

  endssn = NumGames # assume every game is a regular season game

  print NumGames

  for i in range(NumGames):
    dummy_col1 = dummy_games[i].contents[0].contents[0]

    if type(dummy_col1) is bs2.NavigableString:
      
      #if 'POSTSEASON GAME LOG' in dummy_col1:
      if 'REGULAR SEASON GAME LOG' in dummy_col1:  
        print "Success!", i
        #endssn = i
        begssn = i
  try:
      print begssn
  except:
      print "Only %s Games available, skipping player." % NumGames
      return -1
  #print len(games)

  #print endssn

  outfile1 = "data/%s_%s_%s_%d.csv" % (league.upper(),split_title1[0],split_title1[1],year)

  if league == 'nba':
      outfile2 = "Shots_%s_%s_%s_%d.csv" % (league.upper(),split_title1[0],split_title1[1],year)
  elif league == 'nfl':
      outfile2 = "Passes_%s_%s_%s_%d.csv" % (league.upper(),split_title1[0],split_title1[1],year)


  #print outfile1

  # prepare data output
  MyFile1 = open(outfile1,'wb')
  # MyFile2 = open(outfile2,'wb')
  # open an output csv file
  Data1 = csv.writer(MyFile1)
  # Data2 = csv.writer(MyFile2)

  fields = ['ID','NumDate','StDate','Opp','Score','Res','FGA','FGM','FGP','3PA','3PM','3PP','FTA','FTM','FTP','PTS','REB','AST', 'BLK', 'STL', 'PF','TO', 'MIN']

  Data1.writerow(fields)

  count_games = 0

  #for i in range(endssn):
  for i in range(begssn,NumGames):
    #print games[i].contents, type(games[i].contents[0])

    # only process rows where the 1st column is a number (Jersey Number
    # of a player)

    # some 
    if type(games[i].contents[0]) is bs2.Tag:
      col1 = games[i].contents[0].contents[0]
    else:
      col1 = games[i].contents[1].contents[0]
    
    #col1 = games[i].contents[0].contents[0]

    #print col1

    # the date patters are the same for nba and nfl, but
    # different for mlb, so need to take care of this here
    # this means different conditions for extracting the data,
    # which justifies the 'if' statements.

    match = None # nba/nfl date pattern
    if type(col1) is bs2.NavigableString:
      match = re.search('(\w+)\s+(\d+)\/(\d+)',col1)

    if match and len(games[i].contents) > 4:

      count_games = count_games + 1
      
      #print match.group(), match.group(1), match.group(2), match.group(3)

      PlyrData = []
      PlyrData.append(plyr_id)
      if league == 'nba':
        month = int(match.group(2))
        day = int(match.group(3))

        # date appended twice
        PlyrData.append("%d%02d%02d" % (year,month,day))
        PlyrData.append(str(match.group()))

        #print PlyrData
        #print games[i].contents[1].contents[0].contents[2]
        #print len(games[i].contents[1].contents[0])
        
        # this is to deal with the fact that the NJ Nets are no mo...
        # opponent append
        if len(games[i].contents[1].contents[0]) > 2:
          #print games[i].contents[1].contents[0].contents[2].contents[0]
          PlyrData.append(str(games[i].contents[1].contents[0].contents[2].contents[0].contents[0]))
        else:
          #print games[i].contents[1].contents[0].contents[1].contents[0]
          PlyrData.append(str(games[i].contents[1].contents[0].contents[1].contents[0]))

        #PlyrData.append(str(games[i].contents[1].contents[0].contents[2].contents[0].contents[0]))
        #print games[i].contents[2].contents[2].contents[0]
        # score append
        PlyrData.append(str(games[i].contents[2].contents[2].contents[0]))
        #print games[i].contents[2].contents[0].contents[0]
        # W/L append
        PlyrData.append(str(games[i].contents[2].contents[0].contents[0]))
        #print [int(games[i].contents[4].contents[0].split('-')[0]),int(games[i].contents[4].contents[0].split('-')[1])]
        # FGA/FGM append
        PlyrData = PlyrData + [int(games[i].contents[4].contents[0].split('-')[1]), int(games[i].contents[4].contents[0].split('-')[0])]
        # FGP append
        PlyrData.append(float(games[i].contents[5].contents[0]))
        # 3PGA/3PGM append
        PlyrData = PlyrData + [int(games[i].contents[6].contents[0].split('-')[1]), int(games[i].contents[6].contents[0].split('-')[0])]
        # 3PP append
        PlyrData.append(float(games[i].contents[7].contents[0]))     
        # FTA/FTM append
        PlyrData = PlyrData + [int(games[i].contents[8].contents[0].split('-')[1]), int(games[i].contents[8].contents[0].split('-')[0])]
        # FGP append
        PlyrData.append(float(games[i].contents[9].contents[0]))   
        
        # Points
        PlyrData.append(int(games[i].contents[16].contents[0]))   
        
        # REB
        PlyrData.append(int(games[i].contents[10].contents[0]))
        # AST
        PlyrData.append(int(games[i].contents[11].contents[0]))
        # BLK
        PlyrData.append(int(games[i].contents[12].contents[0]))
        # STL
        PlyrData.append(int(games[i].contents[13].contents[0]))
        # PF
        PlyrData.append(int(games[i].contents[14].contents[0]))
        # TO
        PlyrData.append(int(games[i].contents[15].contents[0]))

        # minutes played
        PlyrData.append(int(games[i].contents[3].contents[0]))

        #print PlyrData

      elif league == 'nfl':
        month = int(match.group(2))
        day = int(match.group(3))

        # date, appended twice
        PlyrData.append("%d%02d%02d" % (year,month,day))
        PlyrData.append(str(match.group()))

        #print PlyrData
        PlyrData.append(str(games[i].contents[1].contents[0].contents[2].contents[0].contents[0]))
        #print games[i].contents[1].contents[0].contents[2].contents[0].contents[0]
        PlyrData.append(str(games[i].contents[2].contents[2].contents[0]))
        #print games[i].contents[2].contents[2].contents[0]
        PlyrData.append(str(games[i].contents[2].contents[0].contents[0]))
        #print games[i].contents[2].contents[0].contents[0]
        PlyrData = PlyrData + [int(games[i].contents[4].contents[0]), int(games[i].contents[3].contents[0]), float(games[i].contents[6].contents[0])/100]
        #print [games[i].contents[4].contents[0], games[i].contents[3].contents[0], games[i].contents[6].contents[0]]

        #print PlyrData

      Data1.writerow(PlyrData)    
      # Data2.writerow(PlyrData[5:])

  print "There were %d games!" % count_games

  MyFile1.close()
  # MyFile2.close()

  return 1

def ParsePlyrDataByIdMlb(league,plyr_id,year):

  Months = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

  if plyr_id < 0:
    print "No player info, can't parse data...\n"
    return

  request = Request("http://espn.go.com/%s/player/gamelog/_/id/%d/year/%d" %  (league.lower(),plyr_id,year))

  # Try to read the page and handle exceptions if necessary
  # This is especially true for the off-season
  try:
    response = urlopen(request)
  except HTTPError, e:
    print "\tThe server couldn\'t fulfill the request\n"
    print "\tError code:", e.code
    print "\tSorry, there is no data for this player, in this league and year...\n"
    print "\tPlease check the inputs for accuracy...\n"
  except URLError, e:
    print "\tWe failed to reach a server\n"   

  the_page = response.read()
  pool = bs2.BeautifulSoup(the_page)

  title = pool.findAll("title")

  split_title1 = title[0].contents[0].split()
  split_title2 = title[0].contents[0].split('-')

  plyr_name = split_title1[0] + ' ' + split_title1[1]
  team_name = split_title2[1].strip()

  print "Parsing data:\n"
  print "\tPlayer Name\t: %s" % plyr_name
  print "\tLeague\t\t: %s" % league.upper()
  print "\tTeam\t\t: %s" % team_name
  print "\tYear\t\t: %d" % year

  games = pool.findAll("tr")

  dummy_games = pool.findAll("tr") # will use this to find postseason start/end

  NumGames = len(games)

  print len(games)

  begssn = 0 # assume every game is a regular season game

  for i in range(NumGames):

    if type(dummy_games[i].contents[0]) is bs2.Tag:
      dummy_col1 = dummy_games[i].contents[0].contents[0]
    else:
      dummy_col1 = games[i].contents[1].contents[0]

    if type(dummy_col1) is bs2.NavigableString:
      
      if 'Regular Season Games' in dummy_col1:
        print "Success!", i
        begssn = i

  #print len(games)

  print begssn

  outfile1 = "%s_%s_%s_%d.csv" % (league.upper(),split_title1[0],split_title1[1],year)
  outfile2 = "Batting_%s_%s_%s_%d.csv" % (league.upper(),split_title1[0],split_title1[1],year)

  #print outfile1

  # prepare data output
  MyFile1 = open(outfile1,'wb')
  # MyFile2 = open(outfile2,'wb')
  # open an output csv file
  Data1 = csv.writer(MyFile1)
  # Data2 = csv.writer(MyFile2)

  count_games = 0

  for i in range(begssn,NumGames):

    #print games[i].contents, type(games[i].contents[0])

    # only process rows where the 1st column is a number (Jersey Number
    # of a player)

    # some 
    if type(games[i].contents[0]) is bs2.Tag:
      col1 = games[i].contents[0].contents[0]
    else:
      col1 = games[i].contents[1].contents[0]
    
    #col1 = games[i].contents[0].contents[0]

    # date
    #print col1

    # the date patters are the same for nba and nfl, but
    # different for mlb, so need to take care of this here
    # this means different conditions for extracting the data,
    # which justifies the 'if' statements.

    match = None # mlb date pattern
    if type(col1) is bs2.NavigableString:
      match = re.search('(\w+)\s+(\d+)',col1)

    if match and len(games[i].contents) > 4:

      count_games = count_games + 1
      
      # number of times at bat
      #print games[i].contents[4].contents[0].lower()

      #print match.group(), match.group(1), match.group(2), match.group(3)

      PlyrData = []

      month = Months[str(match.group(1)).lower()]
      day = int(match.group(2))

      PlyrData.append("%d%02d%02d" % (year,month,day))
      PlyrData.append(str(match.group()))

      #print PlyrData

      # date
      #print games[i].contents[1].contents[0]
      #print games[i].contents[2].contents[0]#.contents[2].contents[0].contents[0]
      PlyrData.append(str(games[i].contents[2].contents[0].split()[1]))
      #print games[i].contents[3].contents[2].contents[0]
      PlyrData.append(str(games[i].contents[3].contents[2].contents[0]))
      #print games[i].contents[3].contents[0].contents[0]
      PlyrData.append(str(games[i].contents[3].contents[0].contents[0]))

      if games[i].contents[4].contents[0].lower() == 'did not play':
        #print [0, 0]
        PlyrData = PlyrData + [0, 0, 0]
      else:
        #print [games[i].contents[4].contents[0], games[i].contents[6].contents[0]]
        PlyrData = PlyrData + [int(games[i].contents[4].contents[0]), int(games[i].contents[6].contents[0])]
        PlyrData.append(float(games[i].contents[18].contents[0]))

      #print PlyrData

      Data1.writerow(PlyrData)
      # Data2.writerow(PlyrData[-3:])

  print "There were %d games!" % count_games

  MyFile1.close()
  MyFile2.close()

  return


def main():
  if len(sys.argv) != 4:
    print 'usage: python ParseDataByLeague.py league plyr_id year'
    sys.exit(1)

  league = sys.argv[1]
  plyr_id = int(sys.argv[2])
  year = int(sys.argv[3])
  #outfile = 'test_nba%s%s%s.csv' % (str(season[0]),str(season[1]),'seasonWL')

  print league, plyr_id, year

  #Teams = ParseTeams()

  #ParseRoster('Boston Celtics','nba','Paul Pierce')
  #ParseRoster('New York Giants','nfl','Eli Manning')
  #ParseRoster('Los Angeles Angels','mlb','Albert Pujols')
  #ParseRoster('Washington Capitals','nhl','Joel Ward')


  #ParseRoster('Atlanta Hawks')
  
  #ParsePlyrData('Boston Celtics','nba','Paul Pierce',2011)
  #ParsePlyrData('Miami Heat','nba','LeBron James',2011)
  #ParsePlyrData('New York Giants','nfl','Eli Manning',2011)
  #ParsePlyrData('Los Angeles Angels','mlb','Albert Pujols',2011)
  #ParsePlyrData('Los Angeles Lakers','nba','Kobe Bryant',1999)

  ParsePlyrDataById(league,plyr_id,year)
  
  #DateList = GenSeasonDates(int(year))

  #print DateList
  #print outfile

  t1 = time.clock()
  #ParseAllData(['20100427','20100430','20110430'], outfile)
  #ParseAllData(DateList, outfile)  
  t2 = time.clock()

  print "It ran in", t2-t1, ".\n"

  return

if __name__ == '__main__':
  main()
