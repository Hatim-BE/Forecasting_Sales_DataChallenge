import sys

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info() # this shows where (the file and the line) the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occured in script : [{file_name}] at line : [{line_number}] ==> [{str(error)}]"

    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
# if __name__ == "__main__":
#     try:
#         a = 10 / 0
#     except Exception as e:
#         raise CustomException(e, sys)