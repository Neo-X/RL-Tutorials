/*
 * Word.cpp
 *
 *  Created on: 2015-10-25
 *      Author: gberseth
 */

#include "Word.h"


Word::Word() :
	_the_word("")
{
	// TODO Auto-generated constructor stub
}


Word::Word(std::string the_word) :
	_the_word(the_word)
{
	// TODO Auto-generated constructor stub
}

Word::~Word() {
	// TODO Auto-generated destructor stub
}

void Word::updateWord(std::string word)
{
	this->_the_word = word;
}

std::string Word::getWord()
{
	return this->_the_word;
}
