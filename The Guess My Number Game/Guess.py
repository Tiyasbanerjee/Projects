import os
import random
import pandas as pan



current_dir = os.path.dirname(__file__)
data_path = current_dir + "/data_from_privious_games.csv"


if not os.path.exists(data_path):
    with open(data_path, 'w') as f:
        pass
    temp_data = pan.DataFrame(columns=["nums","reps"])

    list_ = [i for i in range(1,21)]
    temp_data = pan.DataFrame({"nums":list_ , "reps":[0]*20})
    temp_data.to_csv(data_path, index=False)



def fetch_guess(lower_range:int , upper_range:int):
    upper_inxex = upper_range 
    lower_index = lower_range - 1


    reps_data = list(pan.read_csv(data_path)["reps"][lower_index:upper_inxex])


    best_guess_apprence = max(reps_data)

    if best_guess_apprence == 0:
        
        return random.randint(lower_range , upper_range)

    else:
       guess = reps_data.index(best_guess_apprence) + 1 + lower_index

       return guess




def guess_logic_and_control(privious_guess:int = 1 , upper_ramge:int = 20 , try_:int = 0):
      try_ += 1

      if privious_guess == upper_ramge:
             print(f"than sure your_guess is {privious_guess}")             
             return privious_guess , try_
      elif  privious_guess >  upper_ramge :
           print(f"you are doing controdiction !!! ")
           return None , 1

      else:
        computer_Guess = fetch_guess(privious_guess , upper_ramge)
        print(f"My guess is ::::::::::: --->>>> {computer_Guess}")
        print("Am i right ?")
        while True:
            usr_inp = input("plz enter -->> 0 (True)| 1 (if guess < your_num ) | 2 (if your_num < guess) ------>>>")
            
            if usr_inp == "0" :
                return computer_Guess , try_
            elif usr_inp == "1" :
                   return guess_logic_and_control((computer_Guess+1) , upper_ramge , try_)
            elif usr_inp == "2" :
                         return guess_logic_and_control(privious_guess , (computer_Guess-1) , try_)
            else:
                 print(f"plz try again, your input || {usr_inp} || is not valid")
            
                


def save_data(num):
     temp_data = pan.read_csv(data_path)
     temp_data.at[num-1 , "reps"] += 1
     temp_data.to_csv(data_path, index=False)


print("lets play a game, imagine any number between 1-to-20 and i will guess it")
your_score = 0
my_scoore = 0



while True:
     flag = int(input("do you want to play ? | enter 0 (to not play) or 1 (to play) |:::---->>>>>"))
     if flag:
          
          corrent_guess , tried = guess_logic_and_control()

          if corrent_guess == None and tried == 1 :
               print(f"contriduction fine !!! But this is fine free...")
          else:
          
                  my_scoore += int(100/tried)
          
                  data = list(pan.read_csv(data_path)["reps"])
                  total = 1    # to avoid 0 , will added 1
                  for i in data:
                          total += i

                  your_score += 100*(total-data[corrent_guess-1])/total
        
                  your_score = 100*(your_score)/(your_score + my_scoore)
                  my_scoore = 100*(my_scoore)/(your_score + my_scoore)

                  print(f"your scoore is:---> {your_score} \n and my scoore is:---> {my_scoore}")  

                  save_data(corrent_guess)



     else:
          print("Thank you to play, hope you enjoied too.")










