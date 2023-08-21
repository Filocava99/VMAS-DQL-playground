from Action import Action


class North(Action):
    def __init__(self):
        super().__init__("North")

class South(Action):
    def __init__(self):
        super().__init__("South")

class East(Action):
    def __init__(self):
        super().__init__("East")

class West(Action):
    def __init__(self):
        super().__init__("West")

class NorthEast(Action):
    def __init__(self):
        super().__init__("NorthEast")

class NorthWest(Action):
    def __init__(self):
        super().__init__("NorthWest")

class SouthWest(Action):
    def __init__(self):
        super().__init__("SouthWest")

class SouthEast(Action):
    def __init__(self):
        super().__init__("SouthEast")

def CleaningAgentsActions_toSeq():
    return [North(), South(), East(), West(), NorthEast(), NorthWest(), SouthEast(), SouthWest()]