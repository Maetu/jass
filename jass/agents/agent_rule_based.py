# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import pickle

import numpy as np

from jass.game.const import card_strings, card_values, color_masks, color_offset, PUSH
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class AgentRuleBased:
    """
    Agent to act as a player in a match of jass.
    """
    ace = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    king = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    queen = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    ober = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    junge = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    ten = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    card_values = np.array(
        [
            # DA DK DQ DJ D10 D9 D8 D7 D6 HA HK HQ HJ H10 H9 H8 H7 H6 SA SK SQ SJ S10 S9 S8 S7 S6 CA CK CQ CJ C10 C9 C8 C7 C6
            [11, 4, 3, 20, 10, 14, 0, 0, 0, 11, 4, 3, 2, 10, 0, 0, 0, 0, 11, 4, 3, 2, 10, 0, 0, 0, 0, 11, 4, 3, 2, 10, 0, 0, 0, 0],
            [11, 4, 3, 2, 10, 0, 0, 0, 0, 11, 4, 3, 20, 10, 14, 0, 0, 0, 11, 4, 3, 2, 10, 0, 0, 0, 0, 11, 4, 3, 2, 10,
             0, 0, 0, 0],
            [11, 4, 3, 2, 10, 0, 0, 0, 0, 11, 4, 3, 2, 10, 0, 0, 0, 0, 11, 4, 3, 20, 10, 14, 0, 0, 0, 11, 4, 3, 2, 10,
             0, 0, 0, 0],
            [11, 4, 3, 2, 10, 0, 0, 0, 0, 11, 4, 3, 2, 10, 0, 0, 0, 0, 11, 4, 3, 2, 10, 0, 0, 0, 0, 11, 4, 3, 20, 10,
             14, 0, 0, 0],
            [11, 4, 3, 2, 10, 0, 8, 0, 0, 11, 4, 3, 2, 10, 0, 8, 0, 0, 11, 4, 3, 2, 10, 0, 8, 0, 0, 11, 4, 3, 2, 10, 0,
             8, 0, 0],
            [0, 4, 3, 2, 10, 0, 8, 0, 11, 0, 4, 3, 2, 10, 0, 8, 0, 11, 0, 4, 3, 2, 10, 0, 8, 0, 11, 0, 4, 3, 2, 10, 0,
             8, 0, 11]
        ], np.int32)


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
        self._logger.info('Trump request')
        trump = 0
        hand = self._rule.get_valid_cards_from_obs(obs)
        if obs.forehand == -1:
            # if forehand is not yet set, we are the forehand player and can select trump or push
            values = (hand * card_values).sum(axis=1)
            if max(values) > 40:
                trump = list(values).index(max(values))
                return trump
            else:
                return PUSH
        values = (hand * card_values).sum(axis=1)
        trump = list(values).index(max(values))
        return trump

        """
        self._logger.info('Trump request')
        if obs.forehand == -1:
            # if forehand is not yet set, we are the forehand player and can select trump or push
            if self._rng.choice([True, False]):
                self._logger.info('Result: {}'.format(PUSH))
                return PUSH
        # if not push or forehand, select a trump
        result = int(self._rng.integers(low=0, high=MAX_TRUMP, endpoint=True))
        self._logger.info('Result: {}'.format(result))
        return result
        """




    def action_play_card(self, obs: GameObservation) -> int:

        """
        Select randomly a card from the valid cards
        Args:
            obs: The observation of the jass match for the current player
        Returns:
            card to play
        """
        self._logger.info('Card request')
        card = 0
        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        #print(valid_cards, "Valid Cards")
        trump = obs.declared_trump
        nrcardsintrick = obs.nr_cards_in_trick
        currenttrick = obs.current_trick
        cardsplayed = obs.nr_played_cards
        tricks = obs.tricks


        cvg = card_values[trump]
        myvalues = valid_cards * cvg
        maxcard = list(myvalues).index(max(myvalues))
        mincard = 0
        if myvalues.sum() !=0:
         minval = list(myvalues).index(np.min(myvalues[np.nonzero(myvalues)]))
         mincard = minval

        #rules for playing obeabe and undeufe
        if trump >= 4:
            #Einen geschobenen Obenabe oder Undenufe eröffnen Sie,
            #sofern Sie nicht Ass und König resp. Sechs und Sieben von der gleichen Farbe haben,
            #mit der schwächsten Karte der stärksten Farbe.
            #Play highest Card if first
            if currenttrick[0] == -1:
                if myvalues.sum() != 0 and cardsplayed == 0:
                    card = maxcard
                    print(card)
                    return card
            #second player trick
            if currenttrick[1] == -1 and currenttrick[0] > -1:
                if myvalues.sum() != 0:
                    if cvg[currenttrick[0]] < maxcard:
                        card = maxcard
                        return card
                    elif cvg[currenttrick[0]] > maxcard:
                        card = mincard
                        return card
            #third player trick
            #Eröffnet die Vorhand beim Obenanbe oder Undenufe mit einem Ass resp. einem Sechser,
            #dann ist der Partner verpflichtet, den König resp. den Siebner zu spielen, wenn er ihn hat.
            if currenttrick[2] == -1 and currenttrick[1] > -1:
                if myvalues.sum() != 0:
                    if cvg[currenttrick[0]] < cvg[currenttrick[1]] < maxcard:
                        card = maxcard
                        return card
                    #to implement höchste Karte ohne Spieler eins zu überbieten
                    elif cvg[currenttrick[1]] < cvg[currenttrick[0]]:
                        card = self._rng.choice(np.flatnonzero(valid_cards))
                        return card
                    elif cvg[currenttrick[0]] < cvg[currenttrick[1]] > maxcard:
                        card = mincard
                        return card
            #last player trick
            if currenttrick[3] == -1 and currenttrick[2] > -1:
                if myvalues.sum() != 0:
                    vct = cvg[currenttrick[0:1:2]]
                    maxvct = list(vct).index(max(vct))
                    if maxvct == 1:
                        card = maxcard
                        return card
                    elif maxvct !=1 and maxvct > maxcard:
                        card = mincard
                        return card
                    elif maxvct !=1 and maxvct < maxcard:
                        card = maxcard
                        return card

        # rules for playing color
        if trump <= 3:
            #array über die momentanen trumpfkarten
            trumpcards = color_masks[trump]
            #print(trumpcards)
            co = color_offset[trump]
            mytrumpcards = valid_cards * trumpcards
            myvaluestrump = myvalues * mytrumpcards
            #print(mytrumpcards,"mytrumpcards")
            mytrumpcardsi = np.where(mytrumpcards == 1)
            mytrumpcardsi = mytrumpcardsi[0]

            #print(mytrumpcardsi, "indizes")

            #print("Meine Trumpfkarten Werte",myvaluestrump)
            #höchste Trumpfskarte
            highestowntrumpcard = list(myvaluestrump).index(max(myvaluestrump))
            #if highestowntrumpcard != 0:
                #print (highestowntrumpcard, "highestowntrumpcard")
            #falls runde 1
            #print("Es sind",cardsplayed, "Karten gespielt")
            if nrcardsintrick == 0:
                #print("testrunde1")
                #Wenn Sie den Trumpf-Bauer solo haben, dann spielen Sie eine tiefe Karte (6,7, 8 oder 9) Ihrer stärksten Farbe.
                if myvalues.sum() != 0 and obs.nr_tricks == 0 and co + 3 in mytrumpcardsi and np.size(mytrumpcardsi) == 1:
                    #get strongest color
                    strongestcolor = 0
                    max_number_in_color = 0
                    for c in range(4):
                        number_in_color = (valid_cards * color_masks[c]).sum()
                        if number_in_color > max_number_in_color:
                            max_number_in_color = number_in_color
                            strongestcolor = c
                    mycolorcards = color_masks[strongestcolor]
                    mycolorcardsvalid = valid_cards*mycolorcards
                    mycolorvalues = mycolorcardsvalid * myvalues
                    if mycolorvalues.sum() != 0:
                        index = np.where(mycolorvalues == np.min(mycolorvalues[np.nonzero(mycolorvalues)]))
                        card = index[0]
                        #print("test3" ,card, mycolorvalues, obs.nr_tricks,currenttrick,nrcardsintrick,tricks,valid_cards)
                        return card

                #Wenn Sie den Trumpf-Bauer plus einen weiteren Trumpf besitzen, dann spielen den niedrigeren Trumpf.
                if obs.nr_tricks == 0 and [(co + 3)] in mytrumpcardsi and np.size(mytrumpcardsi) == 2:
                    index = np.where(mytrumpcardsi != co+3)
                    card = mytrumpcardsi[index]
                    #print("test2",card)
                    return card
                #Wenn sie das Trumpf-Ass plus zwei weitere Trümpfe haben, wird oftmals nicht das Trumpf-Ass sondern der zweithöchste Trumpf ausgespielt
                if obs.nr_tricks == 0 and [co] in mytrumpcardsi and np.size(mytrumpcardsi) == 3:
                    index = np.max(np.where(mytrumpcardsi != co))
                    card = mytrumpcardsi[index]
                    #print("test4",card)
                    return card
                # erster spieler runde 1, ansonsten spiele höchste trumpfkarte (Ass- nell)

                if myvalues.sum() != 0  and myvaluestrump.sum() != 0 and obs.nr_tricks == 0:
                    card = highestowntrumpcard
                    #print("test", card, valid_cards,mytrumpcards,myvaluestrump,myvalues, obs.nr_tricks,currenttrick,nrcardsintrick,tricks)
                    return card
                #falls nur tiefe Trumpfkarten, spiele die tiefste Trumpfkarte
                if myvalues.sum() != 0 and myvaluestrump.sum() == 0 and mytrumpcards.sum() != 0 and obs.nr_tricks == 0:
                    lowtrumps = np.where(mytrumpcards == 1)
                    card = np.max(lowtrumps)
                    return card
                # falls keine trumpfkarte
                if myvalues.sum() != 0 and myvaluestrump.sum() == 0 and mytrumpcards.sum() == 0 and obs.nr_tricks == 0:
                    #print("test1")
                    kqojmatrix = np.add(np.add(self.king, self.queen),np.add(self.ober, self.junge))
                    kqoj = valid_cards * kqojmatrix
                    #print (kqoj,"kqoj")
                    if kqoj.sum() != 0:
                            #print("test2")
                            strongestcolor = 0
                            max_number_in_color = 0
                            for c in range(4):
                                number_in_color = (kqoj * color_masks[c]).sum()
                                if number_in_color > max_number_in_color:
                                    max_number_in_color = number_in_color
                                    strongestcolor = c
                            #print("test3")
                            mycolorcards = color_masks[strongestcolor]
                            mycolorcardsvalid = kqoj * mycolorcards
                            mycolorvalues = mycolorcardsvalid * myvalues
                            index = np.where(mycolorvalues == np.max(mycolorvalues[np.nonzero(mycolorvalues)]))
                            #print (index, "Index")
                            card = index[0]
                            #print(card, "TEST")
                            return card




        # convert to list and draw a value
        card = self._rng.choice(np.flatnonzero(valid_cards))
        self._logger.info('Played card: {}'.format(card_strings[card]))


        return card
