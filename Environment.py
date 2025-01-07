import pygame
import numpy as np
from Graphics import *
from State import *
import random
state = State()


class Environment():

    def __init__(self, state=None):
        self.state:State = state
        self.train = True

    def pieces(self):
        pieces = {
            1:np.array([[1, 1, 1, 1]]),
            2:np.array([[2, 0, 0],[2, 2, 2]]),
            3:np.array([[0, 0, 3],[3, 3, 3]]),
            4:np.array([[4, 4],[4, 4]]),
            5:np.array([[0, 5, 5],[5, 5, 0]]),
            6:np.array([[0, 6, 0],[6, 6, 6]]),
            7:np.array([[7, 7, 0],[0, 7, 7]])
        }
        return pieces
    
    def select_falling_piece(self, state): #בוחר חלק רנדומלי
        falling_piece_id = state.next_piece # לוקח את החלק שנבחר קודם
        falling_piece_shape = self.pieces()[falling_piece_id] # שומר את הצורה של החלק הנבחר
        falling_piece_col = COLS // 2 - len(falling_piece_shape[0]) // 2 #ממקם אותו באמצע השורה העליונה
        state.falling_piece = 0, falling_piece_col, falling_piece_shape # ממקם את החלק על הלוח
        state.next_piece = random.randint(1, 7) # בוחר חלק הבא חדש

    def add_piece(self, state): # הוספת החלק ללוח  
        state.reward += 0.01 ################################
        row, col, piece = state.falling_piece # מוצא את מיקום וצורת החלק
        rows, cols = piece.shape # מוצא את אורך ורוחב החלק
        # if row + rows <= ROWS:
        state.board[row:row+rows, col:col+cols] += piece # מוסיף את החלק במקום המתאים על הלוח

    def del_piece(self, state): # מחיקת חלק מהלוח
        row, col, piece = state.falling_piece #מוצא את מיקום וצורת החלק
        erase = piece==0 # מוצא את כל הריבועים הריקים מסביב לצורה
        rows, cols = erase.shape # מוצא את אורך ורוחב המערך 
        state.board[row:row+rows, col:col+cols] *= erase # מכפיל את החלק במערך של המחיקה, כך שאיפה שלא היה 0 עכשיו יוכפל ב0 ויימחק

    def down_piece (self, state): # הורדת שורה
        self.del_piece(state)
        state.down() #החלק יורד שורה
        self.add_piece(state)

    def move (self, state, action): # הזזת החלק
        row, col, piece = state.falling_piece

        if not self.no_move(state, action):
            self.del_piece(state) #מחיקת החלק מהלוח
            if action == 0:
                pass # לא קורה כלום
            elif action == 1:
                state.falling_piece = row, col - 1, piece # הזזה שמאלה
            elif action == 2:
                state.falling_piece = row, col + 1, piece # הזזה ימינה
            elif action == 3:
                state.falling_piece = row, col, np.rot90(piece, k=3) # סיבוב
            elif action == 4:
                state.fall_speed = 4 # הגדלת מהירות הנפילה
            elif action == 5:
                state.fall_speed = state.FALL_SPEED #החזרת מהיות הנפילה לרגיל

            self.add_piece(state) # הוספת החלק חזרה ללוח

    def no_move(self, state, action): # בודק אם ,תזוזה אפשרית 
        row, col, piece = state.falling_piece
        is_no_move = False

        if action == 0:
            pass # לא קורה כלום
        elif action == 1:
            is_no_move = self.is_collision(state, falling_piece=state.falling_piece, dCol=-1) # בודקים אם תהיה התנגשות בהזזה שמאלה
        elif action == 2:
            is_no_move = self.is_collision(state, falling_piece=state.falling_piece, dCol=1) # בודקים אם תהיה התנגשות בהזזה ימינה
        elif action == 3:
            falling_piece = row, col, np.rot90(piece, k=3)
            is_no_move = self.is_collision(state, falling_piece=falling_piece) # בודקים אם תהיה התנגשות בסיבוב
        
        return is_no_move

    def is_collision(self, state, falling_piece, dRow = 0, dCol = 0): # בודק אם תהיה התנגשות
        row, col, piece = falling_piece # מציאת מיקום וצורת החלק
        rows, cols = piece.shape # מציאת אורך ורוחב החלק
        if row + rows + dRow > ROWS:
            state.reward += 0.02 ######################################
        if row + rows + dRow > ROWS or col + cols + dCol > COLS or col + dCol < 0: # אם יגיע לרצפה או לאחד הצדדים 
            return True
        # check if will collide
        self.del_piece(state)
        result = (state.board[row+dRow:row+rows+dRow, col+dCol:col+cols+dCol] * piece).sum() # בודק אם במקרה של הזזה סכום מכפלת האזור בלוח והחלק עצמו יהיה גדול מאפס
        self.add_piece(state)
        if result > 0: # אם כן זאת אומרת שכבר יש שם חלק
            return True
        else:
            return False
        
    def reached_top(self, state): # בודק אם חלק הגיע עד למעלה
        row, col, piece = state.falling_piece
        if self.is_collision(state, falling_piece=state.falling_piece, dRow=1) and row == 0: # אם הוא נקתע וגם נמצא בשורה העליונה
            # print(state.score) # מדפיס את הניקוד
            return True # מחזיר אמת - הגיע עד למעלה
        return False
    
    def clear_rows(self, state): # מחיקת שורות מלאות מהלוח
        temp_board = state.board.copy() # יצירת לוח זמני
        temp_board[temp_board != 0] = 1 # החלפת כל המספרים שלא 0 ב1
        row_sums = np.sum(temp_board, axis=1) # מציאת הסכום של כל שורה
        full_rows = np.where(row_sums == 10)[0] # יצירת מערך עם מספרי השורות המלאות - סכום 10
              

        for i in full_rows: # מעבר על מערך זה
            state.board[1:i+1] = state.board[0:i] # הורדת כל השורות בלוח עד השורה המלאה ב1
            state.board[0] = 0 # שם שורה ריקה חדשה למעלה
        
        if len(full_rows) > 0: # אם יש לפחות שורה מלאה אחת
            self.update_score(state, len(full_rows)) # עדכון הניקוד
            if not self.train:
                line_clear = pygame.mixer.Sound('sounds/line_clear.mp3') # ניגון צליל
                line_clear.play()


    def update_score(self, state, num): # עדכון הניקוד
        if num == 1:
            state.score += 40
            state.reward += 0.04
        elif num == 2:
            state.score += 100
            state.reward += 1
        elif num == 3:
            state.score += 300
            state.reward += 3
        elif num == 4:
            state.score += 1200
            state.reward += 12
        

    def reward(self, state):
        if self.reached_top(state):
            state.reward -= 5

        r = state.reward
        # state.reward = 0

        return r
    
    def newState(self):
        state = State()
        self.select_falling_piece(state) # בוחר חלק ראשון
        self.add_piece(state)
        return state
    
    def next_state(self, state, action):
        if action: # אם היא חוקית
            self.move(state, action) # מזיז את החלק
        if not self.is_collision(state, falling_piece=state.falling_piece, dRow=1):
                self.down_piece(state)    # יורד שורה
                state.reward += 0.005 ##########################
        else:
                self.clear_rows(state) # מוחק שורות אם צריך  
                self.select_falling_piece(state) #אתחול המשתנים
                self.add_piece(state)
        return state, self.reward(state)

        

