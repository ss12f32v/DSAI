class env():
    def __init__(self):
        """
        Attributes:
            status: The current status of the stock holding by us
                value = 1, -1 ,0 
        Ter:
            If violate the rule define by TA then terminate the program.
            Just use it for handling unexpected error.
        
        prev_income:
            Record previous day's income to calculate reward by their difference.
        Income:
            Use to calculate the final income.

        """
        self.status = 0
        self.Ter = None
        self.prev_income = 0
        self.income = 0 


    def reset(self):
        """Reset all the parameters """
        self.status = 0
        self.Ter = None
        self.prev_income = 0
        self.income = 0 

    def execute_action(self, action, open_prize, tomorrow_open_prize):
        """Execute the action determined by Agent """
        if action==1:
            if self.status ==1:
                self.Ter = True
                return self.Reward(tomorrow_open_prize, self.Ter), self.Ter
            
            self.income -= open_prize # It means you are using open_prize to buy a stock.
            self.status += 1

            reward = self.Reward(tomorrow_open_prize)
        elif action ==0:
            # Do nothing
            reward = self.Reward(tomorrow_open_prize)
            return reward, self.Ter
            
        
        elif action == -1:
            if self.status ==-1:
                self.Ter = True
                return self.Reward(tomorrow_open_prize, self.Ter), self.Ter
            
            self.income += open_prize # It means you are using open_prize to sell a stock.
            self.status -= 1
            reward = self.Reward(tomorrow_open_prize)
        else:
            raise ExceptionError
        return reward, self.Ter

    def Current(self):
        """Retrun current status and income"""
        return (self.status, self.income)
    
    def Reward(self, tomorrow_open_prize, ter = None):
        """
            Calculate the reward by substraction
            Assume tomorrow is the last day, and get the fake final income
        
        
        """
        
        if ter:
            reward = -500
            return reward
        if self.status ==1:
            reward = tomorrow_open_prize + self.income
        elif self.status == 0:
            reward = self.income
        elif self.status == -1:
            reward = self.income - tomorrow_open_prize
        return reward

    def Epoch_Reward(self, tomorrow_open_prize):
        if self.status ==1:
            income = tomorrow_open_prize - self.income
        elif self.status == 0:
            income = self.income
        elif self.status == -1:
            income = self.income - tomorrow_open_prize
        return income