/*
 * Word.h
 *
 *  Created on: 2015-10-25
 *      Author: gberseth
 */

#ifndef WORD_H_
#define WORD_H_

#include <string>

class Word
{
public:
	Word();
	Word(std::string the_word);
	virtual ~Word();

	virtual void updateWord(std::string word);
	virtual std::string getWord();

private:
	std::string _the_word;
};

#endif /* WORD_H_ */
