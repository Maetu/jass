# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import pickle

import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_observation import GameObservation


class Agent:
    """
    Agent to act as a player in a match of jass.
    """
    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # self._logger.setLevel(logging.INFO)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the match observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.match.const or jass.match.const.PUSH
        """
        # predict trump with sklearn
        # import model
        hand = self._rule.get_valid_cards_from_obs(obs)
        handf = np.append(hand, [1])
        print(handf)
        filename = "finalized_model.sav"
        loaded_model = pickle.load(open(filename, 'rb'))
        if obs.forehand == -1:
            # predict
            predict = loaded_model.predict_proba([handf])
            print(predict, "modelbasedtrump")
            trump = np.argmax(predict)
            print(trump, "trump selected first player")
            if trump == 6:
                return PUSH
            return trump
        predict = loaded_model.predict_proba([handf])
        predict = predict[0]
        predict[6] = 0
        print(predict, "modelbasedtrumpgeschoben")
        trump = np.argmax(predict)
        return trump

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the match observation

        Returns:
            the card to play, int encoded as defined in jass.match.const
        """
        raise NotImplementedError

