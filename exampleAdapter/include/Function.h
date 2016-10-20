/*
 * Function.h
 *
 *  Created on: Oct 19, 2016
 *      Author: Glen
 */

#ifndef EXAMPLEADAPTER_INCLUDE_FUNCTION_H_
#define EXAMPLEADAPTER_INCLUDE_FUNCTION_H_

#include <vector>

class Function {
public:
	Function();
	virtual ~Function();

	virtual double func(std::vector<double> x);
	virtual void setStateBounds(std::vector<std::vector<double>> bounds);
	virtual void setActionBounds(std::vector<std::vector<double>> bounds);

	virtual std::vector<double> norm_state(std::vector<double> state)
	{
	    return _norm_action(state, this->_state_bounds);
	}
	virtual std::vector<double> scale_state(std::vector<double> state)
	{
	    return _scale_action(state, this->_state_bounds);
	}
	virtual std::vector<double> norm_action(std::vector<double> state)
	{
		return _norm_action(state, this->_action_bounds);
	}
	virtual std::vector<double> scale_action(std::vector<double> action)
	{
		return _scale_action(action, this->_action_bounds);
	}

	virtual std::vector<double> _norm_action(std::vector<double> action_, std::vector<std::vector<double>> action_bounds_)
	{
	    /*
	        Normalizes the action
	        Where the middle of the action bounds are mapped to 0
	        upper bound will correspond to 1 and -1 to the lower
	        from environment space to normalized space
	    */
		std::vector<double> avg;
		for (size_t i =0; i < action_.size(); i++)
		{
			avg.push_back((action_bounds_[0][i] + action_bounds_[1][i])/2.0);
		}
	    // avg = (action_bounds_[0] + action_bounds_[1])/2.0;
		std::vector<double> normalized_action;
		for (size_t i =0; i < action_.size(); i++)
		{
			normalized_action.push_back((action_[i] - (avg[i])) / (action_bounds_[1][i]-avg[i]));
		}
	    return normalized_action;
	}

	virtual std::vector<double> _scale_action(std::vector<double> normed_action_, std::vector<std::vector<double>> action_bounds_)
	{
	    /*
	        from normalize space back to environment space
	        Normalizes the action
	        Where 0 in the action will be mapped to the middle of the action bounds
	        1 will correspond to the upper bound and -1 to the lower
	    */
		std::vector<double> avg;
		for (size_t i = 0; i < normed_action_.size(); i++)
		{
			avg.push_back((action_bounds_[0][i] + action_bounds_[1][i])/2.0);
		}
	    // avg = (action_bounds_[0] + action_bounds_[1])/2.0;
		std::vector<double> scaled_action;
		for (size_t i = 0; i < normed_action_.size(); i++)
		{
			scaled_action.push_back(normed_action_[i] * (action_bounds_[1][i] - avg[i]) + avg[i]);
		}
	    return scaled_action;
	}

private:

	std::vector<std::vector<double>> _state_bounds;
	std::vector<std::vector<double>> _action_bounds;

};

#endif /* EXAMPLEADAPTER_INCLUDE_FUNCTION_H_ */
