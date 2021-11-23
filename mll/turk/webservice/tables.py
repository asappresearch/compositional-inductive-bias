import sqlalchemy as sa
from sqlalchemy.orm import declarative_base


Base = declarative_base()


# specific game instance for specific player
class GameInstance(Base):  # type: ignore
    """
    table should be unique for (requester_id, task_id) tuple

    other columns are dependent variables, eg score
    """
    __tablename__ = 'game_instance'

    __table_args__ = (sa.UniqueConstraint('requester_id', 'task_id'),)

    id = sa.Column(sa.Integer, primary_key=True)
    requester_id = sa.Column(sa.String)
    task_id = sa.Column(sa.String)

    example_idx = sa.Column(sa.Integer)
    seed = sa.Column(sa.Integer)
    score = sa.Column(sa.Integer)
    num_steps = sa.Column(sa.Integer)
    max_cards = sa.Column(sa.Integer)
    num_holdout = sa.Column(sa.Integer)
    status = sa.Column(sa.String)
    start_datetime = sa.Column(sa.String)
    finish_datetime = sa.Column(sa.String)
    completion_code = sa.Column(sa.String)
    remote = sa.Column(sa.String)
    feedback = sa.Column(sa.String)
    duration_seconds = sa.Column(sa.Integer)
    num_cards = sa.Column(sa.Integer)
    num_holdout_correct = sa.Column(sa.Integer)
    cents_per_ten = sa.Column(sa.Integer)

    def __repr__(self):
        return f'<GameInstance requester_id={self.requester_id} task_id={self.task_id}>'


# result of single example for specfici game instance of specific player
class StepResult(Base):  # type: ignore
    """
    table should be unique for (requester_id, task_id, example_idx)

    we are going to treat rows in this table as immutable
    """
    __tablename__ = 'step_results'

    __table_args__ = (sa.UniqueConstraint('requester_id', 'task_id', 'example_idx'),)

    id = sa.Column(sa.Integer, primary_key=True)

    requester_id = sa.Column(sa.String)
    task_id = sa.Column(sa.String)
    example_idx = sa.Column(sa.Integer)

    meaning = sa.Column(sa.String)
    image_path = sa.Column(sa.String)
    expected_utt = sa.Column(sa.String)
    start_datetime = sa.Column(sa.String)
    num_cards = sa.Column(sa.Integer)  # decides score
    is_holdout = sa.Column(sa.Boolean)

    status = sa.Column(sa.String)

    finish_datetime = sa.Column(sa.String)
    player_utt = sa.Column(sa.String)
    score = sa.Column(sa.Integer)


def create_tables(engine):
    Base.metadata.create_all(engine)
