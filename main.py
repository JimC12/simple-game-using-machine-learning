'''
Main.py Starting File
'''
import os
import numpy as np
import tensorflow as tf
from model import Model
from plot import Plot
from game import Game
import matplotlib.pyplot as plt


#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.feature_length = 6
        self.label_length = 4

        self.cost_plot = Plot([], 'Step', 'Cost')
        self.accuracy_plot = Plot([], 'Step', 'Accuracy')
        self.checkpoint = 'data/Checkpoints/turn_based_ai.ckpt'

        self.X = tf.placeholder(tf.float32, [None, self.feature_length])
        self.Y = tf.placeholder(tf.float32, [None, self.label_length])
        self.model = Model(self.X, self.Y)
        self.model2 = Model(self.X, self.Y)
        self.global_step = 0
        self.winRatio = np.zeros(100000)

        self.training_data_x = np.empty((0, self.feature_length))
        self.training_data_y = np.empty((0, self.label_length))
        
        
        self.training_data_x_loss = np.empty((0, self.feature_length))
        self.training_data_y_loss = np.empty((0, self.label_length))
        
        self.test_training_data_x = np.empty((0, self.feature_length))
        self.test_training_data_y = np.empty((0, self.label_length))
        
        self.Count1Win = 0

    def add_training_data(self, features, labels, add_to_test_data):
        self.training_data_x = np.concatenate((self.training_data_x, features), axis=0)
        self.training_data_y = np.concatenate((self.training_data_y, labels), axis=0)

        if add_to_test_data:
            self.test_training_data_x = np.concatenate((self.test_training_data_x, features), axis=0)
            self.test_training_data_y = np.concatenate((self.test_training_data_y, labels), axis=0)
            
    def add_training_data_loss(self, features, labels, add_to_test_data):
        self.training_data_x_loss = np.concatenate((self.training_data_x_loss, features), axis=0)
        self.training_data_y_loss = np.concatenate((self.training_data_y_loss, labels), axis=0)

        return


    def get_data_for_prediction(self, user, opponent):
        data = np.array([user.attack / user.max_attack,
                        user.defence / user.max_defence,
                        user.health / user.max_health,
                        opponent.attack / opponent.max_attack,
                        opponent.defence / opponent.max_defence,
                        opponent.health / opponent.max_health
                        ])
        return np.reshape(data, (-1, self.feature_length))

    def start_user_vs_ai(self, restore):
        train = True
        players_turn = True
        player_goes_first = True
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if restore:
                saver.restore(sess, self.checkpoint)

            while train:
                game = Game(player_goes_first, self.feature_length, self.label_length)

                player_goes_first = not player_goes_first

                while not game.game_over:
                    predicted_action = 0

                    if game.players_turn is False:
                        #Predict opponent's action
                        data = self.get_data_for_prediction(game.user, game.opponent)
                        #print('opponents\'s view: {}'.format(data))
                        predicted_actions = sess.run(self.model.prediction, { self.X: data })[0]
                        #predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions))
                        predicted_action = np.argmax(predicted_actions) + 1

                    #Play Game
                    did_player_win = game.run(predicted_action)

                    if game.game_over and did_player_win == None:
                        train = False
                    elif game.game_over:
                        #record winning data
                        if did_player_win:
                            self.add_training_data(game.player_training_data.feature, game.player_training_data.label, False)
                        else:
                            self.add_training_data(game.opponent_training_data.feature, game.opponent_training_data.label, False)

                #Train
                if 1 == 2:# ToDo: put back to train this is to test  )
                    for _ in range(50):
                        
                        training_data_size = np.size(self.training_data_x, 0)
                        random_range = np.arange(training_data_size)
                        np.random.shuffle(random_range)

                        for i in range(training_data_size):
                            random_index = random_range[i]
                            _, loss = sess.run(self.model.optimize, { self.X: np.reshape(self.training_data_x[random_index], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_y[random_index],(-1, 4))})
                        #_, loss = sess.run(model.optimize, { X: self.training_data_x, Y: self.training_data_y })
                        self.global_step += 1

                        current_accuracy = sess.run(self.model.error, { self.X: self.training_data_x, self.Y: self.training_data_y })
                        self.cost_plot.data.append(loss)
                        self.accuracy_plot.data.append(current_accuracy)

                       

                    print('Saving...')
                    saver.save(sess, self.checkpoint)

                    print('Epoch {} - Loss {} - Accuracy {}'.format(self.global_step, loss, current_accuracy))

                    #weights = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_1/weights:0'))[0]
                    
                    #Move out into class
                    # plt.close('all')
                    # plt.figure()
                    # plt.imshow(weights, cmap='Greys_r', interpolation='none')
                    # plt.xlabel('Nodes')
                    # plt.ylabel('Inputs')
                    # plt.show()
                    # plt.close()

                    self.cost_plot.save_sub_plot(self.accuracy_plot,
                    "data/Charts/{} and {}.png".format(self.cost_plot.y_label, self.accuracy_plot.y_label))


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def start_ai_vs_ai(self, restore, number_of_games):
        train = False
        number_of_games_played = 0
        max_turns = 40
        current_turns = 0
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if restore:
                saver.restore(sess, self.checkpoint)

            while number_of_games_played < number_of_games:
                game = Game(True, self.feature_length, self.label_length)
                current_turns = 0
                #game_over = False
                # user = None
                # opponent = None
                train = False

                while not game.game_over:
                    predicted_action = 0
                    #Predict action
                    if game.players_turn is False:
                        data = self.get_data_for_prediction(game.user, game.opponent)
                    else:
                        data = self.get_data_for_prediction(game.opponent, game.user)
                        
                    #print('features view: {}'.format(data))
                    if game.players_turn is False:
                        predicted_actions = sess.run(self.model.prediction, { self.X: data })[0]
                    else:
                        predicted_actions = sess.run(self.model2.prediction, { self.X: data })[0]
                        
                    predicted_actions = self.sigmoid(predicted_actions)
                    MaxPred = np.sum(predicted_actions)
                    ProbPred = predicted_actions / MaxPred
                    # print(ProbPred)
                    # print(MaxPred)
                    # print(predicted_actions)
                    # hi
                    #probabilities = sess.run(tf.nn.softmax(predicted_actions[0]))
                    probabilities = ProbPred
                    choices = np.arange(1, self.label_length + 1)

                    choice = np.random.choice(choices, p=probabilities)   
                    #choice = 1                                     
                    
                    # predicted_action = np.argmax(predicted_actions) + 1
                    # predicted_action = np.random.randint(4) + 1
                    
                    #Play Game
                    did_player_1_win = game.run_ai_game(choice)

                    if game.game_over:
                        #record winning data
                        if did_player_1_win:
                            self.add_training_data(game.player_training_data.feature, game.player_training_data.label, True)
                            self.add_training_data_loss(game.opponent_training_data.feature, 1-game.opponent_training_data.label, True)
                            Play1 = True
                            self.Count1Win += 1
                            #self.add_training_data(game.opponent_training_data.feature, 1 - game.opponent_training_data.label, False)
                        else:
                            self.add_training_data(game.opponent_training_data.feature, game.opponent_training_data.label, True)
                            self.add_training_data_loss(game.player_training_data.feature, 1-game.player_training_data.label, True)
                            
                            Play1 = False
                            #self.add_training_data(game.player_training_data.feature, 1 - game.player_training_data.label, False)

                        number_of_games_played += 1
                        train = True

                    current_turns += 1
                    if current_turns >= max_turns:
                        game.game_over = True
                        train = True
                        print("hit max turns")
                        

                    
                #Train
                if train and np.size(self.training_data_x, 0) > 0:
                    for _ in range(1):
                        
                        training_data_size = np.size(self.training_data_x, 0)
                        training_data_loss_size = np.size(self.training_data_x_loss, 0)
                        self.training_data_y_loss *= 0.33
                        
                        # By taking out the negative on one then it wins more often
                        # worked when all set to 0.25 or 1, but not 0
                        # Still a slight bias for going first you win but player1 doesnt converge to attack atttack attack
                        
                        
                        
                        #print(self.training_data_x)
                        #print(self.training_data_y)
                        #print(self.training_data_x_loss)
                        #print(self.training_data_y_loss)
                        
                        #hi
                        #random_range = np.arange(training_data_size)
                        #np.random.shuffle(random_range)

                        for i in range(training_data_size):
                            random_index = i #random_range[i]
                            if Play1:
                                _, loss = sess.run(self.model.optimize, { self.X: np.reshape(self.training_data_x[random_index], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_y[random_index],(-1, 4))})
                            else:
                                _, loss = sess.run(self.model2.optimize, { self.X: np.reshape(self.training_data_x[random_index], (-1,    self.feature_length)), self.Y: np.reshape(self.training_data_y[random_index],(-1, 4))})
                                
                                                    #_, loss = sess.run(model.optimize, { X: self.training_data_x, Y: self.training_data_y })
                        
                        #random_range_loss = np.arange(training_data_size)
                        #np.random.shuffle(random_range_loss)
                        for i in range(training_data_loss_size):
                            random_index_loss = i #random_range_loss[i]
                            if Play1:
                                _, loss = sess.run(self.model2.optimize, { self.X: np.reshape(self.training_data_x_loss[random_index_loss], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_y_loss[random_index_loss],(-1, 4))})
                            else:
                                _, loss = sess.run(self.model.optimize, { self.X: np.reshape(self.training_data_x_loss[random_index_loss], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_y_loss[random_index_loss],(-1, 4))})
                
                        self.global_step += 1

                    current_accuracy = sess.run(self.model.error, { self.X: self.test_training_data_x, self.Y: self.test_training_data_y })
                    self.cost_plot.data.append(loss)
                    self.accuracy_plot.data.append(current_accuracy)

                    self.training_data_x = np.empty((0, self.feature_length))
                    self.training_data_y = np.empty((0, self.label_length))
                    #self.test_training_data_x = np.empty((0, self.feature_length))
                    #self.test_training_data_y = np.empty((0, self.label_length))

                    
                    self.winRatio[self.global_step] = self.Count1Win / self.global_step
                    #user_input = input('paused')
                    #print('Saving...')
                    #saver.save(sess, self.checkpoint)

                    print('Epoch {} - Loss {} - Accuracy {}'.format(self.global_step, loss, current_accuracy))

                    #weights = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_1/weights:0'))[0]
                    
                    #Move out into class
                    # plt.close('all')
                    # plt.figure()
                    # plt.imshow(weights, cmap='Greys_r', interpolation='none')
                    # plt.xlabel('Nodes')
                    # plt.ylabel('Inputs')
                    # plt.show()
                    # plt.close()
                    if self.global_step%50 ==0:
                        self.cost_plot.save_sub_plot(self.accuracy_plot,
                        "data/Charts/{} and {}.png".format(self.cost_plot.y_label, self.accuracy_plot.y_label))
                        
                        plt.figure()
                        lin = np.zeros(self.global_step)
                        lin += 0.5
                        plt.plot(self.winRatio[0:self.global_step])
                        plt.plot(lin)
                        plt.savefig('data/Charts/winratio.png')
                        plt.close()

                    # user_input = input('Continue (y/n)')
                    # if user_input == 'n':
                    #     number_of_games_played = number_of_games
#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/

#Main().start_user_vs_ai(True)
Main().start_ai_vs_ai(False, 10000)
