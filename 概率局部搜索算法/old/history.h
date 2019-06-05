#ifndef _HISTORY_H_
#define _HISTORY_H_

#include <stdio.h>
#include <stdlib.h>
#include "wsat.h"
#include <cfloat>


//#define MAX_VARS  400010       //4000010
//#define MAX_CLAUSES   3800000      //17000000

#define HLIST_SIZE 10000000  //记录搜索历史的步数
//#define DBL_MAX    100.0

int changeOfA;

int teststep = 0;

int learn_size, last_learn_size;
unsigned long long sum_unsat_num; // = 0
int avg_unsat_num;

int local_min_unsat_num; // = MAX_CLAUSES;
int no_better_step; // = 0
#define MAX_NO_BETTER_STEP 50000
int min_unsat_step; // = 0
float   old_hp[11] = {0.5, 0.33, 0.25, 0.2, 0.17, 0.14, 0.12, 0.11, 0.1, 0.09, 0.08};
float   temp_hp[11] = {0.31,  0.29, 0.25, 0.2, 0.17, 0.14, 0.12, 0.11, 0.1, 0.09, 0.08}; //{0.5, 0.33, 0.25, 0.2, 0.17, 0.14};//
float   hp[11] = {0.5, 0.33, 0.25, 0.2, 0.17, 0.14, 0.12, 0.09, 0.1, 0.09, 0.08};
int		hpn = 8;
float	freshp = hp[hpn];
#define MAX_HPN 10


float cfreshp(int hpn)
{
	return (1.0/(float)(hpn+1));
}

// Define a data structure for a literal in the SAT problem.
struct lit_h {
    int var_num;		//variable num, begin with 1
    int sense;			//is 1 for true literals, 0 for false literals.
};

/*parameters of the instance*/
int     num_vars_h;		//var index from 1 to num_vars_h
int     num_clauses_h;	//clause index from 0 to num_clauses-1
//int		max_clause_len;
//int		min_clause_len;

/* literal arrays */
int*	var_poslit_h[MAX_VARS];				//var_lit[v][j]: the clause number of the clause that the j'th positive literal of v occurs in.
int*	var_neglit_h[MAX_VARS];
int		var_poslit_count_h[MAX_VARS];
int		var_neglit_count_h[MAX_VARS];

lit_h*	clause_lit_h[MAX_CLAUSES];		//clause_lit_h[i][j] means the j'th literal of clause i.
int		clause_lit_count_h[MAX_CLAUSES]; 	// amount of literals in each clause

struct mbcount {
    int clause_num;
    unsigned int	make_count;
    unsigned int	nomake_count;
    unsigned int	break_count;
    unsigned int	nobreak_count;
	
};
int zexingxievalue[MAX_VARS];

struct mbmsg
{
	int break_clause;
	int break_count;
	int make_count;
	float prob;

//	float prob_bm;
};

mbmsg** clause_mb_clause; //mb_var_np[v][neg][pos]
int clause_mb_count[MAX_CLAUSES];

int clause_make_times[MAX_CLAUSES];


int* make_set;
int make_set_count;
int* break_set;
int break_set_count;

//float* var_cf_prob[MAX_VARS];
float var_break_prob[MAX_VARS];


//变元翻转引起子句变化的频次统计
mbcount* var_poslit_mbcount[MAX_VARS];
mbcount* var_neglit_mbcount[MAX_VARS];


//变元翻转序列
int flip_list[HLIST_SIZE];
int flip_list_size;
unsigned int var_flip_times[MAX_VARS] = {0};  //局部变元翻转次数统计
unsigned int clause_change_times[MAX_CLAUSES] = {0};  //局部子句赋值改变频次统计
unsigned int clause_change_times_long[MAX_CLAUSES] = {0}; 
float clause_change_weight[MAX_CLAUSES];
unsigned int max_change_count0 = 0, max_change_count1 = 0;




float clause_break_key_prob[MAX_CLAUSES];  //子句使关键子句不满足概率累加 <权重>  可能大于1



//float clause_make_key_prob[MAX_CLAUSES];
void reset_history_array_0()
{
    int v, c, i, j;
    flip_list_size = 0;
    for(c = 0; c < num_clauses_h; ++c)
    {
        clause_change_times[c] = 0;

		clause_make_times[c] = 0;

		for(i = 0; i < clause_mb_count[c]; ++i)
		{
			clause_mb_clause[c][i].break_count = 0;

			//	clause_mb_clause[c][i].make_count = 0;
		}
    }

    for(v = 1; v <= num_vars_h; ++v)
    {
        var_flip_times[v] = 0;
        for(i = 0; i < var_poslit_count_h[v]; ++i)
        {
            var_poslit_mbcount[v][i].make_count = 0;
            var_poslit_mbcount[v][i].nomake_count = 0;
            var_poslit_mbcount[v][i].break_count = 0;
            var_poslit_mbcount[v][i].nobreak_count = 0;
			
	
        }

    }
}
//局部统计清0
void reset_history_array()
{
    int v, c, i, j;
    flip_list_size = 0;
    for(c = 0; c < num_clauses_h; ++c)
    {
        clause_change_times[c] = 0;

		clause_make_times[c] = 0;
//		clause_break_times[c] = 0;
		for(i = 0; i < clause_mb_count[c]; ++i)
		{
			clause_mb_clause[c][i].break_count = 0;
            //	clause_mb_clause[c][i].make_count = 0;
		}
    }
    for(v = 1; v <= num_vars_h; ++v)
    {
        var_flip_times[v] = 0;
		
        for(i = 0; i < var_poslit_count_h[v]; ++i)
        {
            var_poslit_mbcount[v][i].make_count = 0;
            var_poslit_mbcount[v][i].nomake_count = 0;
            var_poslit_mbcount[v][i].break_count = 0;
            var_poslit_mbcount[v][i].nobreak_count = 0;
        }

        for(i = 0; i < var_neglit_count_h[v]; ++i)
        {
            var_neglit_mbcount[v][i].make_count = 0;
            var_neglit_mbcount[v][i].nomake_count = 0;
            var_neglit_mbcount[v][i].break_count = 0;
            var_neglit_mbcount[v][i].nobreak_count = 0;
        }
    }
}
void init_history_array_0()
{
    int v, c, i;

    //根据大小创建var_poslit_mbcount, var_neglit_mbprob等矩阵  变元编号从1开始
    for (v = 1; v <= num_vars_h; ++v)
    {
    	/*
		var_cf_prob[v] = new float[num_clauses_h];
		for(c = 0; c < num_clauses_h; ++c) var_cf_prob[v][c] = 0;   */
        //all_var_poslit_mbcount[v] = new frequency[var_poslit_count_h[v]+1];
        var_poslit_mbcount[v] = new mbcount[var_poslit_count_h[v]+1];  //计数
        for(i = 0; i < var_poslit_count_h[v]; ++i)
        {
            var_poslit_mbcount[v][i].clause_num = var_poslit_h[v][i];
        }
        var_poslit_mbcount[v][i].clause_num = -1;

        //all_var_neglit_mbcount[v] = new frequency[var_neglit_count_h[v]+1];
        var_neglit_mbcount[v] = new mbcount[var_neglit_count_h[v]+1];  //计数

        for(i = 0; i < var_neglit_count_h[v]; ++i)
        {
            var_neglit_mbcount[v][i].clause_num = var_neglit_h[v][i];
        }
		var_neglit_mbcount[v][i].clause_num = -1;
    }
    //cout << "reset_history_array" << endl;

    reset_history_array_0();
}
void init_history_array()
{
    int v, c, i;

    //根据大小创建var_poslit_mbcount, var_neglit_mbprob等矩阵  变元编号从1开始
    for (v = 1; v <= num_vars_h; ++v)
    {
    	/*
		var_cf_prob[v] = new float[num_clauses_h];
		for(c = 0; c < num_clauses_h; ++c) var_cf_prob[v][c] = 0;   */

        //all_var_poslit_mbcount[v] = new frequency[var_poslit_count_h[v]+1];
        var_poslit_mbcount[v] = new mbcount[var_poslit_count_h[v]+1];  //计数

        for(i = 0; i < var_poslit_count_h[v]; ++i)
        {
            var_poslit_mbcount[v][i].clause_num = var_poslit_h[v][i];
        }
        var_poslit_mbcount[v][i].clause_num = -1;

        //all_var_neglit_mbcount[v] = new frequency[var_neglit_count_h[v]+1];
        var_neglit_mbcount[v] = new mbcount[var_neglit_count_h[v]+1];  //计数

        for(i = 0; i < var_neglit_count_h[v]; ++i)
        {
            var_neglit_mbcount[v][i].clause_num = var_neglit_h[v][i];
        }
		var_neglit_mbcount[v][i].clause_num = -1;
    }
    //cout << "reset_history_array" << endl;

    reset_history_array();
}
struct positive_signratio{int positive;int all;};
void print_solution()
{
     int    i;
    //cout<<"v ";
    for (i=1; i<=num_vars_h; i++) {
		if(cur_soln[i]==0) cout<<"-";
		cout<<i;
		
		if(i%20==0) 
		{
			cout<<endl;
			//cout<<"v ";
		}
		else cout<<' ';
     }
     cout<<endl;
}
void print_solution(positive_signratio *s)
{
     int    i;
	positive_signratio *pre=s;
    for (i=1; i<=num_vars_h; i++) {
		if(cur_soln[i]==0) {cout<<"-";}
		else{(s->positive)+=1;}
                s->all++;
		s++;
		cout<<i;
		
		if(i%20==0) 
		{
			cout<<endl;
			//cout<<"v ";
		}
		else cout<<' ';
     }
     cout<<endl;

     
}
int build_instance_h(char *filename)
{
    char    line[1024];
    char    tempstr1[10];
    char    tempstr2[10];
    int     cur_lit;
    int     i,j,k;
    int	    v,c,c1;

	sum_unsat_num = 0;
	local_min_unsat_num = MAX_CLAUSES;
	no_better_step = 0;
	min_unsat_step = 0;

    ifstream infile(filename);
    if(infile==NULL) {
        cout<<"Invalid filename: "<< filename<<endl;
        return 0;
    }

    /*** build problem data structures of the instance ***/
    string str;
	getline(infile, str, '\n');
	while (str[0] != 'p')
		getline(infile, str, '\n');

	str.copy(line, str.length(), 0);

    sscanf(line, "%s %s %d %d", tempstr1, tempstr2, &num_vars_h, &num_clauses_h);
    //ratio = (double)num_clauses_h/num_vars_h;

	learn_size = num_clauses_h*10;
	last_learn_size = learn_size;
	avg_unsat_num = num_vars_h;
	//last_point_avg_unsat_num = num_vars_h;

    for (c = 0; c < num_clauses_h; c++)
	{
		clause_lit_count_h[c] = 0;
		clause_break_key_prob[c] = 0;
//		clause_make_key_prob[c] = 0;
	}

    for (v=1; v <= num_vars_h; ++v)
    {
        var_poslit_count_h[v] = 0;
        var_neglit_count_h[v] = 0;
    }

    max_clause_len = 0;
    min_clause_len = num_vars_h;

    int *temp_lit = new int [num_vars_h+1];//local

 //   cout << "Now, read the clauses, one at a time." << endl;
    for (c = 0; c < num_clauses_h; c++) 	// 读取一行即一个子句
    {
        infile>>cur_lit;
        while (cur_lit != 0) {
            temp_lit[clause_lit_count_h[c]] = cur_lit;	//
            clause_lit_count_h[c]++;
            infile>>cur_lit;
        }
        clause_lit_h[c] = new lit_h[clause_lit_count_h[c]+1];
        for(i=0; i<clause_lit_count_h[c]; ++i)
        {
            //clause_lit_h[c][i].clause_num = c;
            clause_lit_h[c][i].var_num = abs(temp_lit[i]);
            if (temp_lit[i] > 0)
            {
                clause_lit_h[c][i].sense = 1;
                var_poslit_count_h[clause_lit_h[c][i].var_num]++;
            }
            else
            {
                clause_lit_h[c][i].sense = 0;
                var_neglit_count_h[clause_lit_h[c][i].var_num]++;
            }
        }
        clause_lit_h[c][i].var_num = 0;
        //clause_lit_h[c][i].clause_num = -1;
        if(clause_lit_count_h[c] > max_clause_len)
            max_clause_len = clause_lit_count_h[c];
        else if(clause_lit_count_h[c] < min_clause_len)
            min_clause_len = clause_lit_count_h[c];
    }
	double s;
int tempindex=1;

	while(infile>>s){if(s>(rand()%10)/10.0)zexingxievalue[tempindex++]=0;
		else zexingxievalue[tempindex++]=1;};
	
    infile.close();

    delete[] temp_lit;

	int max_count = 0;
    //creat var literal arrays
    for (v=1; v<=num_vars_h; ++v)
    {
		if(var_poslit_count_h[v] > max_count) max_count = var_poslit_count_h[v];
        var_poslit_h[v] = new int[var_poslit_count_h[v]+1];
        var_poslit_count_h[v] = 0;	//reset to 0, for build up the array

		if(var_neglit_count_h[v] > max_count) max_count = var_neglit_count_h[v];
        var_neglit_h[v] = new int[var_neglit_count_h[v]+1];
        var_neglit_count_h[v] = 0;
    }

	make_set = new int[max_count];
	break_set = new int[max_count];

    //cout << "scan all clauses to build up var literal arrays" << endl;
    for (c = 0; c < num_clauses_h; ++c)
    {
        for(i=0; i<clause_lit_count_h[c]; ++i)
        {
            v = clause_lit_h[c][i].var_num;

            if(clause_lit_h[c][i].sense==1)
            {
                var_poslit_h[v][var_poslit_count_h[v]] = c;
                ++var_poslit_count_h[v];
            }
            else
            {
                var_neglit_h[v][var_neglit_count_h[v]] = c;
                ++var_neglit_count_h[v];
            }
        }
    }

    for (v=1; v <= num_vars_h; ++v)
    {
        //var_poslit_h[v][var_poslit_count_h[v]].var_num = 0;
        var_poslit_h[v][var_poslit_count_h[v]]=-1;

        //var_neglit_h[v][var_neglit_count_h[v]].var_num = 0;
        var_neglit_h[v][var_neglit_count_h[v]]=-1;
    }

	clause_mb_clause = new mbmsg*[num_clauses_h];
    
	bool flag = false;
	int temp_vec[num_clauses_h];
	//cout << "make clause_mb_clause" << endl;
	for(c = 0; c < num_clauses_h; ++c)
	{
		//cout << "c = " << c << endl;
		clause_mb_count[c] = 0;
		for(i = 0; i < clause_lit_count_h[c]; ++i)
		{
			v = clause_lit_h[c][i].var_num;
			if(clause_lit_h[c][i].sense == 1) //正文字
			{
				for(j = 0; j < var_neglit_count_h[v]; ++j)
				{
					c1 = var_neglit_h[v][j];
					flag = false;
					for(k = 0; k < clause_mb_count[c]; ++k)
					{
						if(temp_vec[k] == c1){flag = true;break;}
					}

					if(!flag) temp_vec[clause_mb_count[c]++] = c1;
				}
			}
			else //负文字
			{
				for(j = 0; j < var_poslit_count_h[v]; ++j)
				{
					c1 = var_poslit_h[v][j];
					flag = false;
					for(k = 0; k < clause_mb_count[c]; ++k)
					{
						if(temp_vec[k] == c1){flag = true;break;}
					}

					if(!flag) temp_vec[clause_mb_count[c]++] = c1;
				}
			}
		}
		clause_mb_clause[c] = new mbmsg[clause_mb_count[c]];
		for(i = 0; i < clause_mb_count[c]; ++i)
		{
			clause_mb_clause[c][i].break_clause = temp_vec[i];
			clause_mb_clause[c][i].prob = 0;
			//	clause_mb_clause[c][i].prob_bm = 0;
		}
	}
	//cout << "init_history_array" << endl;
	init_history_array_0();
    return 1;
}
void free_history_array()
{
    int i;
    for (i = 0; i < num_clauses_h; i++)
    {
		delete[] clause_mb_clause[i];
    }

    for(i = 1; i <= num_vars_h; ++i)
    {
        delete[] var_poslit_mbcount[i];
        delete[] var_neglit_mbcount[i];
        delete[] var_poslit_h[i];
        delete[] var_neglit_h[i];
	//	delete[] var_cf_prob[i];
    }

	delete[] clause_mb_clause;
	delete[] make_set;
	delete[] break_set;
}


void compute_prob()
{
    int c, i, j;
	max_change_count0 = max_change_count1;
    max_change_count1 = 0;
	
    for(c = 0; c < num_clauses_h; ++c)
    {
        if(clause_change_times[c] > max_change_count1)
        {
            max_change_count1 = clause_change_times[c];
        }
    }

	if(max_change_count0 == 0) max_change_count0 = max_change_count1; //第一次学习时
    //cout<<"--------------------------"<<endl;
	for(c = 0; c < num_clauses_h; ++c)
    {
		//子句权值计算
        clause_change_weight[c] = ((float)clause_change_times[c])/max_change_count1;
		// cout << "clause_change_weight = " << clause_change_weight[c] << endl;
		//mbprob计算
		if(clause_make_times[c] == 0)
		{
			for(i = 0; i < clause_mb_count[c]; ++i)
			{
				clause_mb_clause[c][i].prob = 0;
			}
		}
/*
	    if(clause_break_times[c] == 0)
		{
			for(i = 0; i < clause_mb_count[c]; ++i)
			{
				clause_mb_clause[c][i].prob_bm = 0;
			}
		}
*/
		//cout<<" ("<<c<<"):";
		if(clause_make_times[c] != 0 )
		{
			
			for(i = 0; i < clause_mb_count[c]; ++i)
			{
				clause_mb_clause[c][i].prob = (double)clause_mb_clause[c][i].break_count/clause_make_times[c];
				//clause_mb_clause[c][i].prob_long = (double)clause_mb_clause[c][i].break_count_long/clause_make_times_long[c];
				//if(clause_mb_clause[c][i].prob-clause_mb_clause[c][i].prob_long>0.1||
				 //     -clause_mb_clause[c][i].prob+clause_mb_clause[c][i].prob_long>0.1 )
				//	  clause_mb_clause[c][i].prob = clause_mb_clause[c][i].prob_long;
				//cout<<clause_mb_clause[c][i].prob_long<<" " ;
				//clause_mb_clause[c][i].prob = clause_mb_clause[c][i].prob+0.1*clause_mb_clause[c][i].prob_long;
                //	clause_mb_clause[c][i].prob_bm = (double)clause_mb_clause[c][i].make_count/clause_break_times[c];
                //  cout << "clause_mb_clause.prob_bm = " << clause_mb_clause[c][i].prob_bm << endl;
			}
		}
    }
	//cout<<endl;
	/*
	//计算变元使子句 make、break概率
    for(v = 1; v <= num_vars_h; ++v)
    {
        for(i = 0; i < var_poslit_count_h[v]; ++i)
        {
            if(var_poslit_mbcount[v][i].break_count == 0) var_poslit_mbprob[v][i].break_prob = 0;
            else var_poslit_mbprob[v][i].break_prob = ((float)var_poslit_mbcount[v][i].break_count)/var_flip_times[v];
        }

        for(i = 0; i < var_neglit_count_h[v]; ++i)
        {
            if(var_neglit_mbcount[v][i].break_count == 0) var_neglit_mbprob[v][i].break_prob = 0;
            else var_neglit_mbprob[v][i].break_prob = ((float)var_neglit_mbcount[v][i].break_count)/var_flip_times[v];
        }
    }*/

	int c1, c2;
	float temp_sum;
	float temp_sum_bm;
	for(c1 = 0; c1 < num_clauses_h; ++c1)
	{
		temp_sum = 0;
		for(j = 0; j < clause_mb_count[c1]; ++j)
		{
			c2 = clause_mb_clause[c1][j].break_clause;
			temp_sum += clause_mb_clause[c1][j].prob*clause_change_weight[c2];
//			temp_sum_bm += clause_mb_clause[c1][j].prob_bm * clause_change_weight[c2];
		}
		//if(clause_break_key_prob[c1] == 0) clause_break_key_prob[c1] = temp_sum;
		clause_break_key_prob[c1] = clause_break_key_prob[c1]*(1-freshp) + temp_sum*freshp;
//		clause_make_key_prob[c1] = clause_make_key_prob[c1] * (1-freshp) + temp_sum_bm * freshp;
		//if(clause_break_key_prob[c1] > 0.1)cout<<clause_break_key_prob[c1]<<" ";
	}
	
	//var_cf_prob[v][c];
	/*
	//float break_prob;
	for(v = 1; v <= num_vars_h; ++v)
	{
		var_break_prob[v] = 0;
		for(c = 0; c < num_clauses_h; ++c)
		{
			var_cf_prob[v][c] *= 1-freshp;

			for(i = 0; i < var_poslit_count_h[v]; ++i) //遍历v的正字子句
			{
				//v翻转使子句c不满足的概率
				if(var_poslit_mbcount[v][i].break_count == 0) continue; //var_poslit_mbprob[v][i].break_prob = 0;
				//else var_poslit_mbprob[v][i].break_prob = ((float)var_poslit_mbcount[v][i].break_count)/var_flip_times[v];

				//为了使c满足 使其它子句不满足的传导概率
				c1 = var_poslit[v][i];
				temp_sum = 0;
				for(j = 0; j < clause_mb_count[c1]; ++j)
				{
					c2 = clause_mb_clause[c1][j].break_clause;
					for(k = 0; k < clause_lit_count[c2]; ++k)
					{//保证v不在c2中
						if(clause_lit[c2][k].var_num == v)break;
					}

					if(k == clause_lit_count[c2])
					{
						temp_sum += clause_mb_clause[c1][j].prob*clause_change_weight[c2];
					}
				}
				var_cf_prob[v][c] += freshp*temp_sum*((float)var_poslit_mbcount[v][i].break_count)/var_flip_times[v];
			}

			for(i = 0; i < var_neglit_count_h[v]; ++i) //遍历v的负字子句
			{
				//v翻转使子句c不满足的概率
				if(var_neglit_mbcount[v][i].break_count == 0) continue; //var_poslit_mbprob[v][i].break_prob = 0;
				//else var_poslit_mbprob[v][i].break_prob = ((float)var_poslit_mbcount[v][i].break_count)/var_flip_times[v];

				//为了使c满足 使其它子句不满足的传导概率
				c1 = var_neglit[v][i];
				temp_sum = 0;
				for(j = 0; j < clause_mb_count[c1]; ++j)
				{
					c2 = clause_mb_clause[c1][j].break_clause;
					for(k = 0; k < clause_lit_count[c2]; ++k)
					{//保证v不在c2中
						if(clause_lit[c2][k].var_num == v)break;
					}

					if(k == clause_lit_count[c2])
					{
						temp_sum += clause_mb_clause[c1][j].prob*clause_change_weight[c2];
					}
				}
				var_cf_prob[v][c] += freshp*temp_sum*((float)var_poslit_mbcount[v][i].break_count)/var_flip_times[v];
			}
			if(var_cf_prob[v][c] > 0)
				cout<<var_cf_prob[v][c]<<" ";
		}
	}*/
}

void save_when_flip()
{
	if(step == 0) 
	{
		reset_history_array();  //统计信息清0
		hpn = 0;

	//  for MBWsat1	

		freshp = cfreshp(hpn);

	}

	//记录翻转序列与翻转频次
    flip_list[flip_list_size++] = cur_soln[flipvar]? flipvar:-flipvar;  //flipvar;
    var_flip_times[flipvar]++;

	//计算不满足子句总数
	sum_unsat_num += unsat_stack_fill_pointer;
    //cout<<unsat_stack_fill_pointer<<" ";
    int c, i, j, k, c1, c2; //truep指向正值数组 falsep指向负值数组

	make_set_count = 0;
	break_set_count = 0;
    if(cur_soln[flipvar] == 0) //负字
    {
        for(i = 0; i < var_poslit_count_h[flipvar]; ++i)
        {
            c = var_poslit_h[flipvar][i];
            if(sat_count[c] == 0) //记录make的子句
            {
                clause_change_times[c]++;
				
                var_poslit_mbcount[flipvar][i].make_count++;
		

				make_set[make_set_count++] = c;
				clause_make_times[c]++;
				
            }
            else
            {
                var_poslit_mbcount[flipvar][i].nomake_count++;
	
            }
        }

        for(i = 0; i < var_neglit_count_h[flipvar]; ++i)
        {
            c = var_neglit_h[flipvar][i];
            if(sat_count[c] == 1) //记录break的子句
            {
                clause_change_times[c]++;
		
                var_neglit_mbcount[flipvar][i].break_count++;
   
				break_set[break_set_count++] = c;
//				clause_break_times[c]++;
            }
            else
            {
                var_neglit_mbcount[flipvar][i].nobreak_count++;
		
            }
        }
    }
    else //正字
    {
        for(i = 0; i < var_neglit_count_h[flipvar]; ++i)
        {
            c = var_neglit_h[flipvar][i];
            if(sat_count[c] == 0) //记录make的子句
            {
                clause_change_times[c]++;
			
                var_neglit_mbcount[flipvar][i].make_count++;
			

				make_set[make_set_count++] = c;
				clause_make_times[c]++;
				
            }
            else
            {
                var_neglit_mbcount[flipvar][i].nomake_count++;
		
            }
        }

        for(i = 0; i < var_poslit_count_h[flipvar]; ++i)
        {
            c = var_poslit_h[flipvar][i];
            if(sat_count[c] == 1) //记录break的子句
            {
                clause_change_times[c]++;
			
                var_poslit_mbcount[flipvar][i].break_count++;


				break_set[break_set_count++] = c;
//				clause_break_times[c]++;
            }
            else
            {
                var_poslit_mbcount[flipvar][i].nobreak_count++;
	
            }
        }
    }

	for(i = 0; i < make_set_count; ++i)
	{
		c1 = make_set[i];
		for(j = 0; j < break_set_count; ++j)
		{
			c2 = break_set[j];
			for(k = 0; k < clause_mb_count[c1]; ++k)
			{
				if(clause_mb_clause[c1][k].break_clause == c2)
				{
					clause_mb_clause[c1][k].break_count++;
	
				}
			}
		}
	}


/*
	for(i = 0; i < break_set_count; ++i)
	{
		c1 = break_set[i];
		for(j = 0; j < make_set_count; ++j)
		{
			c2 = make_set[j];
			for(k = 0; k < clause_mb_count[c1]; ++k)
			{
				if(clause_mb_clause[c1][k].break_clause == c2)
				{
					clause_mb_clause[c1][k].make_count++;
				}
			}
		}
	}

*/
    if(flip_list_size == learn_size)
    {
		//cout<<unsat_stack_fill_pointer<<" ";
		bool isJump = false;
		int temp = sum_unsat_num/learn_size;
		//cout<<step<<":"<<temp<<" "<<hpn<<endl;
		
		/*if(learn_size>100&&unsat_stack_fill_pointer==1||unsat_stack_fill_pointer==2)
			learn_size/=10;*/
		
		if(temp < avg_unsat_num )  //&& hpn < MAX_HPN)
		{//下降期
			//cout<<"1";

			//for MBWsat1
			//freshp = hp[++hpn];

			//for MBWsat2
			freshp = cfreshp(++hpn);
		}

		if(temp > avg_unsat_num && hpn > 0)
		{//跳坑期
			//cout<<"2";


			//for MBWsat1
		//	freshp = hp[--hpn];


			//for MBWsat2
			freshp = cfreshp(--hpn);
			isJump = true;
		}

		avg_unsat_num = temp;
		sum_unsat_num = 0;

        compute_prob();  //统计信息转化为概率知识

		if(isJump)
		{
			//动态调整学习步长
			//if(changed_var_count1 > changed_var_count0 && learn_size > 10*num_clauses)
			/*
			if(max_change_count1*last_learn_size > max_change_count0*learn_size && learn_size > 10*num_clauses)
			{
				last_learn_size = learn_size;
				learn_size -= 10*num_clauses;
			}
			else if(max_change_count1*last_learn_size < max_change_count0*learn_size && learn_size < 100*num_clauses)
			{
				last_learn_size = learn_size;
				learn_size += 10*num_clauses;
			}
			else
			{
				last_learn_size = learn_size;
			}
			
			cout<<"learn_size:"<<learn_size/num_clauses<<endl;*/
		}

        reset_history_array();  //统计信息清0
    }
}

/*
float k1 = 0.4;
float k2 = 1;
float ir1 = 1/k2;
float ir2 = 1/(2*k2*k2);
inline float distribut_N(float x)
{
	return k1*ir1*exp(-x*x*ir2);
}
*/


int pick_inference(int clause_num)
{
    int		i,c,v,ci;
    int		*truep, *falsep;
	float	lit_break_key_prob[1000];
	float   sum_prob = 0;
	float	max_prob = 0;


 //   cout << "pick_inference" << endl;

    c = clause_num;
	if(clause_lit_count_h[c] == 1) return clause_lit_h[c][0].var_num; //单子句直接返回

    for(i = 0; i < clause_lit_count_h[c]; ++i)
    {
        v = clause_lit_h[c][i].var_num;
        lit_break_key_prob[i] = 0; //清0 lit_break_key_prob

		/////MAKE
		falsep = (cur_soln[v])? var_neglit_h[v]:var_poslit_h[v];
		for(; (ci = *falsep) != -1; ++falsep)
        {
            if (sat_count[ci] == 0) //make子句 使key值子句变假的概率求和
            {
            	
                lit_break_key_prob[i] += clause_break_key_prob[ci];
                
            }
        }
		/////

        truep = (cur_soln[v])? var_poslit_h[v]:var_neglit_h[v];

        for(; (ci=*truep) != -1; ++truep)
        {
            if (sat_count[ci] == 1) //break子句 使key值子句变假的概率求和
            {
                
                lit_break_key_prob[i] -= clause_break_key_prob[ci];
        
               
            }
        }
     //   cout << "before exp lit_break_key_prob = " << lit_break_key_prob[i] << endl;

		lit_break_key_prob[i] = exp(lit_break_key_prob[i]);
      //  cout << "after exp lit_break_key_prob = " << lit_break_key_prob[i] << endl;

        sum_prob += lit_break_key_prob[i];


    }
	
/*
    if(++teststep == 1000)
    {
        cout << "max sum_prob = " << sum_prob << endl;
        teststep = 0;
    } 
 */

 //   cout << "max sum_prob = " << sum_prob << endl;

    float prob = (rand()%MY_RAND_MAX_INT)*BASIC_SCALE*sum_prob;  //(0到sum_prob的随机数)

 //   cout << "random prob = " << prob << endl;

    sum_prob = 0;
    for(i = 0; i < clause_lit_count_h[c]; ++i)
    {
        sum_prob += lit_break_key_prob[i];

     //   cout << "i = " << i << ", sum_prob = " << sum_prob << endl;

        if(prob <= sum_prob)
        {
     //       cout<<i<<":"<<clause_lit_h[c][i].var_num<<endl;
            return clause_lit_h[c][i].var_num;
        }
    }
	cout<<"???"<<endl;
 return clause_lit_h[c][rand()%clause_lit_count_h[c]].var_num;
}

int pick_bestvar(int* vec, int n)
{
    int		i,c,v,ci;
    int		*truep, *falsep;
	int 	bestvar_array[1000];
    int		bestvar_count;
    double	best_prob = DBL_MAX;
	float	lit_break_key_prob[1000];

    for(i = 0; i < n; ++i)
    {
        v = vec[i];
        lit_break_key_prob[i] = 0; //清0 lit_break_key_prob

		////MAKE
		falsep = (cur_soln[v])? var_neglit_h[v]:var_poslit_h[v];
		for(; (ci = *falsep) != -1; ++falsep)
        {
            if (sat_count[ci] == 0) //make子句 使key值子句变假的概率求和
            {
                lit_break_key_prob[i] -= clause_break_key_prob[ci];

                              
            }
        }
		////

        truep = (cur_soln[v])? var_poslit_h[v]:var_neglit_h[v];

        for(; (ci=*truep) != -1; ++truep)
        {
            if (sat_count[ci] == 1) //break子句 使key值子句变假的概率求和
            {
                
                lit_break_key_prob[i] += clause_break_key_prob[ci];
                               
            }
        }

		if(best_prob > lit_break_key_prob[i])
		{
			best_prob = lit_break_key_prob[i];
			bestvar_array[0] = v;
			bestvar_count = 1;
		}
		else if(best_prob == lit_break_key_prob[i])
		{
			bestvar_array[bestvar_count] = v;
			bestvar_count++;
		}
    }

	return bestvar_array[rand()%bestvar_count];
}


//initiation of the algorithm
void init()
{
	int 		v,c;
	int			i,j;
	int			clause;
	
	//init solution   
	for (v = 1; v <= num_vars_h; v++) {
       cur_soln[v] = zexingxievalue[v];
	   if(changeOfA!=1){
	   if(rand()%2==1) {cur_soln[v] = 1;}
		else {cur_soln[v] = 0;}
	   }
      // cout<< cur_soln[v];
	}
       // printf("\n");

	/* figure out sat_count, and init unsat_stack */
	unsat_stack_fill_pointer = 0;
	for (c=0; c<num_clauses_h; ++c) 
	{
		sat_count[c] = 0;
		
		for(j=0; j<clause_lit_count_h[c]; ++j)
		{
			if (cur_soln[clause_lit_h[c][j].var_num] == clause_lit_h[c][j].sense)
			{
				sat_count[c]++;
				sat_var[c] = clause_lit_h[c][j].var_num;	
			}
		}

		if (sat_count[c] == 0) unsat(c);
	}

	/*calculate the break values*/
	int * truep, * falsep;
	for (v=1; v<=num_vars_h; v++) 
	{
		bbreak[v] = 0;
		
		if (cur_soln[v]==1)
		{
			for (truep=var_poslit_h[v]; (c=*truep)!=-1; truep++)
			{
				if (sat_count[c]==1) bbreak[v]++;
			}	
		}
		else
		{
			for (falsep=var_neglit_h[v]; (c=*falsep)!=-1; falsep++)
			{
				if (sat_count[c]==1) bbreak[v]++;
			}	
		}
	}

}

int verify_sol()
{
	int c,j; 
	int flag;
	
	for (c = 0; c<num_clauses_h; ++c) 
	{
		flag = 0;
		for(j=0; j<clause_lit_count_h[c]; ++j)
			if (cur_soln[clause_lit_h[c][j].var_num] == clause_lit_h[c][j].sense) {flag = 1; break;}

		if(flag ==0){//output the clause unsatisfied by the solution
			cout<<"clause "<<c<<" is not satisfied"<<endl;
			
			for(j=0; j<clause_lit_count_h[c]; ++j)
			{
				if(clause_lit_h[c][j].sense==0)cout<<"-";
				cout<<clause_lit_h[c][j].var_num<<" ";
			}
			cout<<endl;
			
			for(j=0; j<clause_lit_count_h[c]; ++j)
				cout<<cur_soln[clause_lit_h[c][j].var_num]<<" ";
			cout<<endl;

			return 0;
		}
	}
	return 1;
}

void flip_simp(int flipvar)
{
	int c;
	int * truep, * falsep;
	
	cur_soln[flipvar] = 1 - cur_soln[flipvar];

	if (cur_soln[flipvar]==1) {truep=var_poslit_h[flipvar]; falsep=var_neglit_h[flipvar];}
	else {truep = var_neglit_h[flipvar]; falsep = var_poslit_h[flipvar];}

	for (; (c=*truep)!=-1; truep++)
	{
		++sat_count[c];
		if (sat_count[c] == 1) sat(c); // sat_count from 0 to 1
	}

	for (; (c=*falsep)!=-1; falsep++)
	{
		--sat_count[c];
		if (sat_count[c] == 0) unsat(c); //	last_unsatvar[c]=flipvar;
	}
}


#endif
