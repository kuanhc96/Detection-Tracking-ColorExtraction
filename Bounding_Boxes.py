class Bounding_Boxes:
    def __init__(self, ID, frame_id, xl, yl, xr, yr):
        self.id = ID
        self.frame_id = frame_id
        self.xl = xl
        self.yl = yl
        self.xr = xr
        self.yr = yr
        self.red = red
        self.green = green
        self.blue = blue
        

    @property
    def get_details(self):
        return tuple((self.id, self.frame_id, tuple((self.xl, self.yl, self.xr, self.yr)), tuple((self.red, self.green, self.blue))))

    def __repr__(self):
        return "ID: '{}', Frame ID: '{}', Bounding Box: '{}', RGB: '{}'"\
            .format(self.id, self.frame_id, tuple((self.xl, self.yl, self.xr, self.yr)), tuple((self.red, self.green, self.blue)))
