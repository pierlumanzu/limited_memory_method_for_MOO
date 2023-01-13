from progressbar import ProgressBar


class ProgressBarWrapper:

    def __init__(self, max_value: int):
        self.__bar = ProgressBar(max_value=max_value)
        self.__current_value = 0

    def increment_current_value(self):
        self.__current_value += 1

    def show_bar(self):
        self.__bar.update(self.__current_value)
