/*********************************** WalkSAT2013 ************************************
** WalkSAT2013, is a fast implementation version of the WalkSAT algorithm.           **
** It is implemented by Shaowei Cai in 2013.                                         **
** You can find me on www.shaoweicai.net, or email to shaoweicai.cs@gmail.com.       **
**************************************************************************************/
#ifndef _WSAT_H_
#define _WSAT_H_

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <string>
//#include <io.h>

using namespace std;

/* limits on the size of the problem. */
#define MAX_VARS    400010
#define MAX_CLAUSES 3800000

#define pop(stack) stack[--stack ## _fill_pointer]
#define push(item, stack) stack[stack ## _fill_pointer++] = item

int		max_clause_len;		// 最大子句长度
int		min_clause_len;		// 最小子句长度
double	ratio;				// 子句变元比
/* Information about the variables. */
int		bbreak[MAX_VARS];			//break(x), the smaller the better

/* Information about the clauses */
int             sat_count[MAX_CLAUSES];			
int		sat_var[MAX_CLAUSES];

//unsat clauses stack
int		unsat_stack[MAX_CLAUSES];		// 不满足子句集合
int		unsat_stack_fill_pointer;		// 
int		index_in_unsat_stack[MAX_CLAUSES];//which position is a clause in the unsat_stack

/* Information about solution */
bool     cur_soln[MAX_VARS];	//当前赋值

//cutoff
int		max_tries = 10;  //100000;	// 最大尝试次数
unsigned int   	max_flips = 400000000;//4000000000u;	// 每次最大翻转次数
unsigned int	 step;	// 翻转步号
int flipvar;
int min_unsat;


const int   	  MY_RAND_MAX_INT =  10000; // 10000000;
const float 	BASIC_SCALE = 0.0001; //0.0000001; //1.0f/MY_RAND_MAX_FLOAT;

double 		wp; // 噪音参数



inline
void unsat(int clause)
{
	index_in_unsat_stack[clause] = unsat_stack_fill_pointer;
	push(clause,unsat_stack);
}




inline
void sat(int clause)
{
	int index,last_unsat_clause;

	//since the clause is satisfied, its position can be reused to store the last_unsat_clause
	last_unsat_clause = pop(unsat_stack);
	index = index_in_unsat_stack[clause];
	unsat_stack[index] = last_unsat_clause;
	index_in_unsat_stack[last_unsat_clause] = index;
}






#endif

