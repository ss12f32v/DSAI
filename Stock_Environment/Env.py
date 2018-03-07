class env():
    def __init__(self):
        """
        Attributes:
            status: The current status of the stock holding by us
                value = 1, -1 ,0 
        Ter:
            If violate the rule define by TA then terminate the program.
            Just use it for handling unexpected error.
            
        Income:
            Use to calculate the final income.

        """
        open = 0
        self.status = 0
        self.Ter = None
        self.income = 0 
    def execute_action(self, action, open_prize):
            if action==1:
                if self.status ==1:
                    self.Ter = True
                    raise TerminateError
                self.status += 1
                self.income -= open_prize # It means you are using open_prize to buy a stock.
                
            elif action ==0:
                # Do nothing
                pass
            
            elif action == -1:
                if self.status ==-1:
                    self.Ter = True
                    raise TerminateError
                self.status -= 1
                self.income += open_prize # It means you are using open_prize to sell a stock.
            else:
                raise ExceptionError
    def Current(self):
        return (self.status, self.income)