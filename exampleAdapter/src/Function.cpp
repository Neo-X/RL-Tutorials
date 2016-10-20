/*
 * Function.cpp
 *
 *  Created on: Oct 19, 2016
 *      Author: Glen
 */

#include "Function.h"

Function::Function() {
	// TODO Auto-generated constructor stub

}

Function::~Function() {
	// TODO Auto-generated destructor stub
}

double Function::func(std::vector<double> x)
{
	return (std::cos(x[0])-0.75)*(std::sin(x[0])+0.75);
}

void Function::setStateBounds(std::vector<std::vector<double>> bounds)
{
	this->_state_bounds = bounds;
}

void Function::setActionBounds(std::vector<std::vector<double>> bounds)
{
	this->_action_bounds = bounds;
}
