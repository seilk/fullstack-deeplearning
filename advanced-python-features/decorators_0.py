from attrs import define, field


@define
class Newjeans():
    members: list = field(default=['민지','해린', '혜인', '다니엘', '하니'])
    ent: str = field(default="hive")
    tier: int = field(default=0) 
    songs: list = field(default=[])
    
    def set_song(self, song):
        self.songs.append(song)
    
    @classmethod
    def set_tier(cls, tier):
        cls.tier = tier
    
    @property
    def say(self):
        print("둘, 셋! 안녕하세요, NewJeans입니다!")
        


def main():
    idol = Newjeans()
    Newjeans.say
    idol.set_tier(1)

if __name__ == "__main__":
    main()
    
    