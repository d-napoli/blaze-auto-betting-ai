from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from datetime import date, datetime
import time
import random

import math
from shutil import copyfile
import os

# Machine Learning Libs
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, linear_model
from sklearn.model_selection import cross_validate, train_test_split

class Blaze:
    def __init__(self, username, pw):
        self.SITE_LINK = "https://blaze.com"
        self.CRASH_URL = "/en/games/crash"
        self.DIR_PATH = "C:\\Users\\diego\\Documents\\GitHub\\blaze-selenium\\"

        self.FOLLOW_ROBOT_LEAD = False

        self.INITIAL_MONEY = 0
        self.MAX_LOSS = 10

        self.MIN_ROUNDS_PREDICTED = 15
        self.rounds_predicted = 0
        
        self.MINIMUM_FOR_INVESTING = 50

        self.WORTH_IT = 2

        self.BETTING_REAL_MONEY = False

        self.CLASSIFICATION_MODEL = "deep_learning"
        
        self.CSV_HISTORY = "crash_history.csv"
        self.CSV_REAL_TIME = "real_time_analysis.csv"
        self.CSV_MONEY_HISTORY = "money_history.csv"
        self.CSV_ACCURACY_HISTORY = "accuracy_history.csv"

        self.COOLDOWN = 900

        self.username = username
        self.pw = pw

        self.matches_history = []
        self.limit_matches_history = 10

        self.robot_accuracy = 0
        self.robot_real_prediction = 0

        self.against_robot_accuracy = 100
        self.against_robot_real_prediction = 0
        self.against_robot_total_predictions = 0
        self.against_robot_right_predictions = 0

        self.model = Sequential()
        self.clf = neighbors.KNeighborsClassifier()

        self.file = "real_time_analysis.csv"
        self.new_file = "prediction_model.csv"

        self.SLEEPS = {
            "tiny_tiny": 1,
            "tiny": 2,
            "small": 5,
            "medium": 10,
            "big": 30
        }

        self.CRASHES_X_TIME_IN_SECONDS = {
            "1.10X": 2,
            "1.20X": 4,
            "1.40X": 5,
            "1.50X": 6,
            "2.00X": 10,
            "4.00X": 20
        }

        self.SITE_MAP = {
            "labels": {
                "bet_amount": {
                    "xpath": "/html/body/div[1]/main/div[1]/div[3]/div[2]/div[1]/div/div/div[2]/div[1]/div[2]/span"
                },
                "players_bet": {
                    "xpath": "/html/body/div[1]/main/div[1]/div[3]/div[2]/div[1]/div/div/div[2]/div[1]/div[1]/span[1]"
                },
                "last_crash": {
                    "xpath": "/html/body/div[1]/main/div[1]/div[3]/div[2]/div[1]/div/div/div[1]/div[2]/div[2]/div[2]/div[1]/span[$$CRASH_NUMBER$$]"
                },
                "overall_result": {
                    "xpath": "/html/body/div[1]/main/div[1]/div[4]/div[2]/div[1]/div/div/div[2]/div[2]/table/tbody/tr[11]/td[3]/div/span"
                }
            },
            "buttons": {
                "confirm_login": {
                    "xpath": "/html/body/div[1]/main/div[3]/div/div[2]/div[2]/form/div[4]/button"
                },
                "modal_bets_history": {
                    "xpath": "/html/body/div[1]/main/div[1]/div[4]/div[2]/div[1]/div/div/div[1]/div[2]/div[2]/div[2]/div[2]"
                },
                "total_money": {
                    "xpath": "/html/body/div[1]/main/div[1]/div[1]/div/div[2]/div/div/div/div[3]/div/a/div/div/div"
                },
                "bets_history_next_page": {
                    "xpath": "/html/body/div[1]/main/div[3]/div/div[2]/div[2]/div/ul/li[$$PAGE_NUMBER$$]/a"
                },
                "bets_history_close_modal": {
                    "xpath": "/html/body/div[1]/main/div[3]/div/div[1]/i"
                },
                "enter_round": {
                    "xpath": "/html/body/div[1]/main/div[1]/div[3]/div[2]/div[1]/div/div/div[1]/div[1]/div[1]/div[2]/div[2]/button",
                    "get_out_bet_text": "CASHOUT"
                }
            },
            "inputs": {
                "login": {
                    "email": {
                        "xpath": "/html/body/div[1]/main/div[3]/div/div[2]/div[2]/form/div[1]/div/input"
                    },
                    "password": {
                        "xpath": "/html/body/div[1]/main/div[3]/div/div[2]/div[2]/form/div[2]/div/input"
                    }
                },
                "betting": {
                    "bet_value": {
                        "xpath": "/html/body/div[1]/main/div[1]/div[3]/div[2]/div[1]/div/div/div[1]/div[1]/div[1]/div[2]/div[1]/div[1]/div/div[1]/input",
                    }
                }
            },
            "tables": {
                "bets_history": {
                    "time": {
                        "xpath": "/html/body/div[1]/main/div[3]/div/div[2]/div[2]/table/tbody/tr[$$LINE_NUMBER$$]/td[1]/div"
                    },
                    "crashed": {
                        "xpath": "/html/body/div[1]/main/div[3]/div/div[2]/div[2]/table/tbody/tr[$$LINE_NUMBER$$]/td[2]/div"
                    },
                    "initial_line": 2,
                    "max_line": 8,
                    "initial_page": 2,
                    "max_page": 10
                }
            }
        }

        self.CSV_TIME_MAP = {
            "morning": {
                "file_name": "real_time_analysis_morning.csv",
                "time": {
                    "from": 10,
                    "to": 18
                }
            },
            "night": {
                "file_name": "real_time_analysis_night.csv",
                "time": {
                    "from": 19,
                    "to": 0
                }
            },
            "noon": {
                "file_name": "real_time_analysis_noon.csv",
                "time": {
                    "from": 1,
                    "to": 10
                }
            }
        }

        self.driver = webdriver.Chrome(executable_path='C:\\WebDrivers\\chromedriver.exe')
        self.driver.maximize_window()

        self.normalize_file(filter_time_period = False)
        self.stack_rounds_file(filter_time_period = False)
    
    def Login(self):
        self.driver.get(self.SITE_LINK + "?modal=auth&tab=login")

        time.sleep(self.SLEEPS['tiny'])
        self.driver.find_element_by_xpath(self.SITE_MAP['inputs']['login']['email']['xpath']).send_keys(self.username) # username
        time.sleep(self.SLEEPS['tiny_tiny'])
        self.driver.find_element_by_xpath(self.SITE_MAP['inputs']['login']['password']['xpath']).send_keys(self.pw) # password

        time.sleep(120)

    def loop_get_bets_history(self):
        max_loop = 10
        i = 0

        while i < max_loop:
            self.get_bets_history()
            print("I'll hibernate for {} minutes for getting more data - {}".format(str(self.COOLDOWN / 60), datetime.now()))
            time.sleep(self.COOLDOWN)
            i += 1
    
    def get_bets_history(self):
        # Function that opens the blaze history of bets and results
        # and scrappes through the results to store in a csv file
        time.sleep(self.SLEEPS['small'])
        self.driver.get(self.SITE_LINK + self.CRASH_URL)

        time.sleep(self.SLEEPS['tiny'])
        self.driver.find_element_by_xpath(self.SITE_MAP['buttons']['modal_bets_history']['xpath']).click() # opening the table
        time.sleep(self.SLEEPS['small'])

        i = self.SITE_MAP['tables']['bets_history']['initial_page']
        max_page = self.SITE_MAP['tables']['bets_history']['max_page']

        csv_string = ""

        while i < max_page:
            j = self.SITE_MAP['tables']['bets_history']['initial_line']
            max_line = self.SITE_MAP['tables']['bets_history']['max_line']

            while j <= max_line:
                tempo_atual = self.SITE_MAP['tables']['bets_history']['time']['xpath'].replace("$$LINE_NUMBER$$", str(j))
                crashed_atual = self.SITE_MAP['tables']['bets_history']['crashed']['xpath'].replace("$$LINE_NUMBER$$", str(j))
                j += 1

                csv_string += "\n" + self.driver.find_element_by_xpath(tempo_atual).text + ";" + self.driver.find_element_by_xpath(crashed_atual).text

            time.sleep(self.SLEEPS['small'])
            self.driver.find_element_by_xpath(self.SITE_MAP['buttons']['bets_history_next_page']['xpath'].replace("$$PAGE_NUMBER$$", str(i))).click()
            print("I've clicked on the next page {} - {}".format(str(i + 1), datetime.now()))
            time.sleep(self.SLEEPS['small'])

            i += 1

        self.add_new_csv_data(csv_string, self.CSV_HISTORY)
        self.driver.find_element_by_xpath(self.SITE_MAP['buttons']['bets_history_close_modal']['xpath']).click()

    def register_matches_history(self, crash):
        # method to store a defined crash value into the matches history
        if len(self.matches_history) > self.limit_matches_history:
            self.matches_history.pop(0)

        self.matches_history.append(crash)

    def real_time_analysis(self):

        self.INITIAL_MONEY = float(self.get_initial_money()) # get the initial money on the blaze platform

        predictions_right = 0 # number of right prections made by the robot
        
        # predict if the next round is worth it
        # being worth it, it means that the prediction said that the next round
        # will crash over a X amount
        next_round_is_worth_it = 0 

        self.rounds_predicted = 0
        self.against_robot_total_predictions = 0

        i = 1
        max_rounds = 10000 # number maximum of rounds that the robot can run

        csv_string = ""
        accuracy_csv_string = ""

        # the robot has to refresh the screen after a X amount of rounds
        # otherwise the blaze platform will bug
        refresh_at = random.randint(30, 40)

        print("My screen refresh will be at {}".format(str(refresh_at)))

        while i < max_rounds:
            last_crash = self.driver.find_element_by_xpath(self.SITE_MAP['labels']['last_crash']['xpath'].replace("$$CRASH_NUMBER$$", str(1))).text # last crash value
            bet_amount = self.driver.find_element_by_xpath(self.SITE_MAP['labels']['bet_amount']['xpath']).text # total value of the last bet
            players_bet = self.driver.find_element_by_xpath(self.SITE_MAP['labels']['players_bet']['xpath']).text # number of players that betted

            new_round = False

            print("\n** Predictions Right {} - Predicted Rounds {} - Time {}".format(str(predictions_right), str(self.rounds_predicted), str(datetime.now())))

            is_it_worth_it = 1 if float(last_crash.replace("X", "")) > self.WORTH_IT else 0

            csv_string += "\n" + str(last_crash).replace("X", "") + ";" + str(players_bet) + ";" + str(bet_amount).replace("R$", "") + ";" + str(int(datetime.timestamp(datetime.now()))) + ";" + str(datetime.now()) + ";" + str(is_it_worth_it)

            csv_accuracy = self.against_robot_accuracy if self.against_robot_accuracy >= self.robot_accuracy and self.against_robot_accuracy < 100 else self.robot_accuracy
            accuracy_csv_string += "\n" + str(csv_accuracy) + ";" + str(datetime.now())

            self.get_few_last_rounds()

            print("\n\n" + str(self.matches_history) + "\n\n")

            if(len(self.matches_history) > self.limit_matches_history): # the predictions begin only after a few matches, so the robot has a minimum of data to feed the ai model
                next_round_is_worth_it = self.predict_round(self.matches_history)

                if self.robot_real_prediction == 1:
                    self.rounds_predicted += 1

                if self.against_robot_real_prediction == 1:
                    self.against_robot_total_predictions += 1

                if next_round_is_worth_it == 1:
                    if self.BETTING_REAL_MONEY:
                        if not self.round_has_began():
                            self.get_into_next_round()

                print("\n**Next round is worth it {} \n".format(str(next_round_is_worth_it)))
            else:
                print("Predictions starts in {} rounds".format(str(self.limit_matches_history - len(self.matches_history))))

            while not new_round:
                if(last_crash != self.driver.find_element_by_xpath(self.SITE_MAP['labels']['last_crash']['xpath'].replace("$$CRASH_NUMBER$$", str(1))).text):
                    new_round = True
                    time.sleep(self.SLEEPS['tiny_tiny'])

                    checking_last_crash = self.driver.find_element_by_xpath(self.SITE_MAP['labels']['last_crash']['xpath'].replace("$$CRASH_NUMBER$$", str(1))).text

                    if (float(checking_last_crash.replace("X", "")) > self.WORTH_IT and self.robot_real_prediction == 1):
                        if(len(self.matches_history) > self.limit_matches_history):
                            predictions_right += 1
                            next_round_is_worth_it = 0

                    if (float(checking_last_crash.replace("X", "")) > self.WORTH_IT and self.against_robot_real_prediction == 1):
                        if(len(self.matches_history) > self.limit_matches_history):
                            self.against_robot_right_predictions += 1
                            next_round_is_worth_it = 0

                    if self.rounds_predicted > 0 and predictions_right > 0:
                        self.robot_accuracy = (predictions_right * 100) / self.rounds_predicted

                        self.BETTING_REAL_MONEY = False

                        if (self.rounds_predicted >= self.MIN_ROUNDS_PREDICTED) and (self.robot_accuracy >= self.MINIMUM_FOR_INVESTING or self.against_robot_accuracy >= self.MINIMUM_FOR_INVESTING):
                            self.BETTING_REAL_MONEY = True

                        print("\n\n***Accuracy: {}".format(str(self.robot_accuracy)))

                        if not self.FOLLOW_ROBOT_LEAD and self.against_robot_total_predictions > 0:
                            self.against_robot_accuracy = (self.against_robot_right_predictions * 100) / self.against_robot_total_predictions
                            print("\n\n**Against the robot Accuracy: {}".format(str(self.against_robot_accuracy)))
                            print("Against the robot predictions Right {} - Predicted Rounds {} - Time {} **\n\n".format(str(self.against_robot_right_predictions), str(self.against_robot_total_predictions), str(datetime.now())))

            if(i % refresh_at == 0):
                self.add_new_csv_data(csv_string, self.CSV_REAL_TIME)
                self.add_new_csv_data(accuracy_csv_string, self.CSV_ACCURACY_HISTORY)
                csv_string = ""
                self.driver.get(self.SITE_LINK + self.CRASH_URL)
                time.sleep(self.SLEEPS['medium'])
                time.sleep(self.SLEEPS['small'])
                print("Refresh page at {} - after {} executions".format(str(datetime.now()), str(i)))
            i += 1

    def add_new_csv_data(self, content, csv_file):
        time.sleep(self.SLEEPS['tiny'])
        with open(csv_file, "a") as file_object:
            file_object.write(content)
        time.sleep(self.SLEEPS['tiny'])

    def get_initial_money(self):
        time.sleep(self.SLEEPS['tiny'])
        money = self.driver.find_element_by_xpath(self.SITE_MAP['buttons']['total_money']['xpath']).text
        return money.replace("R$", "") # currency in Brazilian reais

    def Store_Money(self):
        money = self.get_initial_money()
        date = datetime.now()

        self.add_new_csv_data(str(money) + ";" + str(date) + "\n", self.CSV_MONEY_HISTORY)

    def machine_learning(self):
        # predictions for the next round uses two different algorithms
        # the first one is machine learning
        predict_column = "Worth_It" # column from the csv file

        df = pd.read_csv('prediction_model.csv', sep=";")
        df.drop(['Crashed_At'], 1, inplace = True) # we drop the column 'crashed_at' as it is the column that we are trying to predict

        X = np.array(df.drop([predict_column], 1)) 
        y = np.array(df[predict_column])

        # layers of the model
        self.model.add(Dense(12, input_dim=self.limit_matches_history + 1, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fitting the model
        self.model.fit(X, y, epochs = 150, batch_size = 10) # before it was 150 epochs, change that later

        _, accuracy = self.model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy*100))

    def k_nearest_neighbors(self):
        # the second algorithm is the k nearest neighbors
        # using both algorithms answer we get the prediction for the next round
        self.CLASSIFICATION_MODEL = "knn"

        predict_column = "Worth_It"

        df = pd.read_csv('prediction_model.csv', sep=";")
        df.drop(['Crashed_At'], 1, inplace = True)

        X = np.array(df.drop([predict_column], 1))
        y = np.array(df[predict_column])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1)

        self.clf.fit(X_train, y_train)

        accuracy = self.clf.score(X_test, y_test)
        print("Accuracy K Nearest Neighbors {}".format(accuracy))

    def predict_round(self, input_data):
        # during tests and developing
        # created this variable 'Follow Robot Lead'
        # if the robot said something we would follow
        # if this variable is set to false, the final action would be the opposite of what the robot said
        self.FOLLOW_ROBOT_LEAD = True

        count = 0
        gratter_than = 0
        return_value = 0
        temp_limit = 4

        limit = temp_limit if len(self.matches_history) > temp_limit else len(self.matches_history) - 1
        
        while count < limit: # max number of > 2 crashes in sequel
            if self.matches_history[count] >= self.WORTH_IT:
                gratter_than += 1
            count += 1

        if gratter_than <= limit:
            # deep learning prediction
            predictions = self.model.predict_classes([input_data])
            return_value_1 = predictions[0][0]

            # knn prediction
            example_measures = np.array([input_data])
            prediction = self.clf.predict(example_measures)
            return_value_2 = prediction[0]

            if return_value_1 == return_value_2: # if both results are equal, than we follow the robot lead
                return_value = return_value_1
            else: # if the results weren't equal, than we will force a negative answer, that's a cautious way of avoiding losing rounds
                print("Results weren't equal, forced the 0 result - KNN {} - Deep Learning {}".format(str(prediction[0]), str(predictions[0][0])))
                return_value = 0

            self.robot_real_prediction = return_value
            self.against_robot_real_prediction = 0

            if self.robot_accuracy <= self.MINIMUM_FOR_INVESTING and self.rounds_predicted >= (self.MIN_ROUNDS_PREDICTED - 5):
                self.FOLLOW_ROBOT_LEAD = False

            if self.FOLLOW_ROBOT_LEAD:
                return return_value
            else:
                print("\n\n** Against the robot prediction, the right result would be: {} **\n\n".format(return_value))
                if return_value == 0:
                    self.against_robot_real_prediction = 1
                    return 1
                else:
                    self.against_robot_real_prediction = 0
                    return 0
        else:
            print("\n\nForced the 0 result - Limit {} - Count {} - GratterThan {}\n\n".format(str(limit), str(count), str(gratter_than)))
            return 0

    def get_few_last_rounds(self):
        initial_number = self.limit_matches_history + 1
        self.matches_history = []
        while initial_number >= 1:
            last_crash = self.driver.find_element_by_xpath(self.SITE_MAP['labels']['last_crash']['xpath'].replace("$$CRASH_NUMBER$$", str(initial_number))).text
            self.register_matches_history(float(str(last_crash).replace("X", "")))
            initial_number -= 1

    def round_has_began(self):   
        time.sleep(2)
        players_bet = self.driver.find_element_by_xpath(self.SITE_MAP['labels']['players_bet']['xpath']).text # get the numbers of players that has beted money
        time.sleep(2)

        # if the number of players that have beted hasn't changed after X seconds, it means that the round has already began
        # if that number didn't change yet, it means that the round hasn't began yet

        if players_bet != self.driver.find_element_by_xpath(self.SITE_MAP['labels']['players_bet']['xpath']).text: # the round hasn't began already, we can still bet the money
            print("Round hasn't began yet")
            return False
        else:
            print("Round has already began")
            return True

    def get_into_next_round(self, percent_cash = 1):
        minimum_money = self.INITIAL_MONEY - ( (self.MAX_LOSS * self.INITIAL_MONEY) / 100 )

        if float(self.get_initial_money()) <= minimum_money:
            print("We have lost a lot of money, better stop")
            return False

        last_crash = self.driver.find_element_by_xpath(self.SITE_MAP['labels']['last_crash']['xpath'].replace("$$CRASH_NUMBER$$", str(1))).text

        remove_money_text = "CASHOUT"
        remove_money_pt_text = "RETIRAR"

        cash = float(self.get_initial_money())

        if cash > 1:
            beting = (percent_cash * cash) / 100 # the money that is going to be bet, is a percentage of the total money available

            if beting < 1:
                beting = 1

            print("I'm going to bet R${}".format(str(beting))) # prints out the amount of money that is going to be bet

            self.driver.find_element_by_xpath(self.SITE_MAP['inputs']['betting']['bet_value']['xpath']).clear()
            self.driver.find_element_by_xpath(self.SITE_MAP['inputs']['betting']['bet_value']['xpath']).send_keys(str(beting))
            self.driver.find_element_by_xpath(self.SITE_MAP['buttons']['enter_round']['xpath']).click()

            max_timeout = 20
            tentatives = 0
            beting_begin = False

            while not beting_begin and tentatives < max_timeout: # keeps waiting for 5 seconds tops
                if remove_money_text in self.driver.find_element_by_xpath(self.SITE_MAP['buttons']['enter_round']['xpath']).text.upper():
                    beting_begin = True

                if remove_money_pt_text in self.driver.find_element_by_xpath(self.SITE_MAP['buttons']['enter_round']['xpath']).text.upper():
                    beting_begin = True

                time.sleep(0.2)
                tentatives += 1

            print("Awaiting time for cashout the mine {}".format(beting))

            money_cooldown = "2.00X" if self.FOLLOW_ROBOT_LEAD else "2.00X" # we expect to remove the money at the crash value of 2.00

            total_sleep_cooldown = self.CRASHES_X_TIME_IN_SECONDS[money_cooldown] # like 11 seconds or something

            inside_loop_cooldown = 0.2
            max_timeout = total_sleep_cooldown * (1 / inside_loop_cooldown)
            tentatives = 0
            go_on_loop = True

            while go_on_loop and tentatives < max_timeout:
                # if the last crash defined up in the code is different than the money showed in the screen
                # it means that we already lost the round by some reason
                # no need for the robot keep waiting inside the loop function
                if last_crash != self.driver.find_element_by_xpath(self.SITE_MAP['labels']['last_crash']['xpath'].replace("$$CRASH_NUMBER$$", str(1))).text:
                    go_on_loop = False # we have already lost our money
                time.sleep(inside_loop_cooldown)
                tentatives += 1

            if last_crash == self.driver.find_element_by_xpath(self.SITE_MAP['labels']['last_crash']['xpath'].replace("$$CRASH_NUMBER$$", str(1))).text:
                self.driver.find_element_by_xpath(self.SITE_MAP['buttons']['enter_round']['xpath']).click()
                print("Clicked for getting money back")
            else:
                print("We've lost R${}".format(beting))

            self.Store_Money()
        else:
            print("Not enough money to bet")
            return False
        
    def get_output_file_name(self, hour = 999):
        if hour == 999:
            hour = int(str(datetime.now().time()).split(":")[0])

        morning = [10, 11, 12, 13, 14, 15, 16, 17, 18]
        night = [19, 20, 21, 22, 23, 0]
        noon = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        if hour in morning:
            index = "morning"
        if hour in night:
            index = "night"
        if hour in noon:
            index = "noon"

        return self.CSV_TIME_MAP[index]['file_name']

    def normalize_file(self, filter_time_period = False):
        self.backup_files()
        i = 0

        new_file_ = ""
        new_file_ = self.get_output_file_name() if filter_time_period else self.new_file

        self.delete_file(new_file_)

        with open(self.file) as openfileobject:
            for line in openfileobject:
                # line = file_content.readline()
                horario = line.split(";")[4].replace("\n", "")

                conteudo_nova_linha = ""

                if horario not in "Time\n":
                    # horario = datetime.strptime(horario.split(".")[0], "%Y-%m-%d %H:%M:%S")
                    # timestamp = datetime.timestamp(horario)

                    linha = line.replace("\n", "")
                    linha = linha.split(";")

                    is_it_worth_it = 1 if float(linha[0]) > self.WORTH_IT else 0

                    if filter_time_period:
                        hora_linha = int(str(linha[4].split(" ")[1]).split(":")[0])

                        if self.get_output_file_name(hora_linha) == new_file_:
                            conteudo_nova_linha = str(float(linha[0])) + ";" + linha[1] + ";" + str(float(linha[2])) + ";" + str(linha[3]) + ";" + linha[4] + ";" + str(is_it_worth_it) + "\n"    
                    else:
                        conteudo_nova_linha = str(float(linha[0])) + ";" + linha[1] + ";" + str(float(linha[2])) + ";" + str(linha[3]) + ";" + linha[4] + ";" + str(is_it_worth_it) + "\n"

                    # print(conteudo_nova_linha, is_it_worth_it)
                    # print(conteudo_nova_linha)

                    if conteudo_nova_linha != "":
                        with open(new_file_, "a") as file_object:
                            file_object.write(conteudo_nova_linha)

                i += 1

        # file_content.close()

    def stack_rounds_file(self, filter_time_period = False):
        self.backup_files()

        i = 0
        j = 0

        self.delete_file(self.new_file)

        original_file = ""
        original_file = self.get_output_file_name() if filter_time_period else self.file

        matches = []
        columns = "Worth_It;Crashed_At;"

        file_content = open(self.file)

        while j <= self.limit_matches_history:
            columns += "Last_Crashed_" + str(self.limit_matches_history - j) + ";"
            j += 1

        with open(self.new_file, "a") as file_object:
            file_object.write(columns[:-1] + "\n")

        with open(original_file) as openfileobject:
            for line in openfileobject:
                line = file_content.readline().replace("\n", "").split(";")

                csv_string = ""

                if i != 0:
                    if len(matches) > self.limit_matches_history:
                        matches.pop(0)

                    matches.append(line[0])

                    if len(matches) > self.limit_matches_history and line[0] != "":
                        csv_string += str(1) if float(line[0]) >= self.WORTH_IT else str(0)
                        csv_string += ";" + str(line[0]) + ";"

                        for m in matches:
                            csv_string += str(m) + ";" 

                        with open(self.new_file, "a") as file_object:
                            file_object.write(csv_string[:-1] + "\n")
                i += 1

    def backup_files(self):
        try:
            copyfile(self.DIR_PATH + self.file, self.DIR_PATH + "Old\\real-time-analysis\\" + self.file.replace(".csv", "") + "_backup" + "_" +  str(datetime.now().year) + "_" + str(datetime.now().month) + "_" + str(datetime.now().day) + "_" + str(datetime.now().time()).replace(":", "_").split(".")[0] + ".csv")
            copyfile(self.DIR_PATH + self.new_file, self.DIR_PATH + "Old\\prediction-model\\" + self.new_file.replace(".csv", "") + "_backup" + "_" +  str(datetime.now().year) + "_" + str(datetime.now().month) + "_" + str(datetime.now().day) + "_" + str(datetime.now().time()).replace(":", "_").split(".")[0] + ".csv")
        except Exception as e:
            pass

    def delete_file(self, file):
        try:
            os.remove(file)
        except Exception as e:
            pass  

b = Blaze("email", "password")
b.machine_learning()
b.k_nearest_neighbors()
b.Login()
b.Store_Money()
b.real_time_analysis()