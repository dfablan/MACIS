#define find_roots_cpp
#include "macis/doping/fix_mu.hpp"

namespace macis {

  double Mu_vs_n(double x, void * params)
  {

    struct impurity_params *p = static_cast<impurity_params*> (params);

    double nel_target = *(p->nel_target);
    size_t norb = *(p->norb);
    size_t n_imp = *(p->n_imp);

    // double delta_CFS = *(p->delta_CFS);
    // if (delta_CFS != 0.0 && nbands != 2)
    // {
        // std::cout << "Error in Mu_Cost_f! delta_CFS is not zero, but nbands is not 2. This is not supported." << std::endl;
        // delta_CFS = 0.0; // Reset to zero to avoid issues
    // }

    std::string ci_exp = *(p->ci_exp);

    // Solve the impurity problem
    double mu = x;  
    double curr_nel_per_spin = 0.0;
    std::vector<double> occs;

    // if( nbands == 2 && delta_CFS != 0.0 )
    // {
        // // If delta_CFS is not zero, we need to account for the Crystal Field Splitting (CFS)
        // for(int i = 0; i < nimp; i++) 
        // {
            // if( i / nsites == 0 )
            // {
                // // For the first site of each band, we add mu
                // p -> ints -> update(i,i,mu-delta_CFS/2.0);
            // }
            // else if( i / nsites == 1 )
            // {
                // // For the second site of each band, we add mu + delta_CFS
                // p -> ints -> update(i,i,mu + delta_CFS/2.0);
            // }
            // else
            // {
                // std::cout << "Error in Mu_vs_n! Invalid index for impurity orbital: i / nsites = " << i / nsites << std::endl;
                // throw( std::runtime_error( "Error in Mu_vs_n! Invalid index for impurity orbital" ) );
            // }
        // }
    // }
    // else
    {
        // If delta_CFS is zero, we just update the diagonal elements with mu
        for(int i = 0; i < n_imp; i++)
        {
            p->T->at(i*norb+i) = mu;
        }
    }

    double E;
    if (ci_exp == "ED")
    {
        E = SolveImpurityED(p);
    }
    else 
    {
        E = SolveImpurityASCI_rot(p);
    }

    occs = *(p->occs);
    *(p->E) = E;

    std::cout << "the current value of mu is " << mu << std::endl;
    std::cout << "the detailed values of the occupations are " << std::endl;
    for (int i = 0; i < n_imp; i++)
    {
        std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }


    curr_nel_per_spin = std::accumulate(occs.begin(), occs.begin()+n_imp, 0.0);
    //curr_nel = 2.0*curr_nel/nimp;
    curr_nel_per_spin = curr_nel_per_spin/n_imp;

    return curr_nel_per_spin;

  }

  double Mu_Cost_f (double x, void * params)
  {

// CAN I AVOID DEFINING THIS HERE? AND USING DIRECTLY SOMETHING LIKE params->norb?
//  struct impurity_params *p = (struct impurity_params *)params;
    struct impurity_params *p = static_cast<impurity_params*> (params);



    size_t norb = *(p->norb);
    size_t n_imp = *(p->n_imp);
    double nel_target = *(p->nel_target);
    
    std::string ci_exp = *(p->ci_exp);
    
    // Solve the impurity problem
    double mu = x;  
    double curr_nel_per_spin = 0.0;



    for(int i = 0; i < n_imp; i++) 
    {
     p->T->at(i*norb+i) = mu;
    }



    // std::vector<double> * T_ptr = p->T;
    // for(int i = 0; i < norb; i++) {
    //     std::cout << "T_diagonal_" << i << " = " << T_ptr->at(i+norb*i) << std::endl;
    // }

    // CAN I AVOID DEFINING THIS HERE? 
    std::vector<double> occs = *(p->occs);
    // curr_nel_per_spin = std::accumulate(occs.begin(), occs.begin()+n_imp, 0.0);
    // std::cout<< "Total number of electrons = "<< 2*curr_nel_per_spin << std::endl;

// IMPLEMENT DIRECT CHOICE FROM INPUT FILE
    double E;
    if (ci_exp == "CAS")
    {
        E = SolveImpurityED(p);
    }
    else 
    {
        E = SolveImpurityASCI_rot(p);
    }

    occs = *(p->occs);
    *(p->E) = E;

    curr_nel_per_spin = std::accumulate(occs.begin(), occs.begin()+*(p->n_imp), 0.0);

    // std::cout<< "Total number of electrons = "<< curr_nel << std::endl;
    // std::cout<< "Goal number of electrons = "<< nel_target << std::endl;

    double err = curr_nel_per_spin - nel_target;

    // std::cout<< "err = "<< err << std::endl;
    // std::cout << "x = " << x << std::endl;

    return err;
  }

  double Mu_Cost_df(double x, void * params)
  {

    struct impurity_params *p = static_cast<impurity_params*> (params);
    //  struct impurity_params *p = (struct impurity_params *)params;
     double mu = x;
     double dstep = *p->dstep;
     double dmu = mu + dstep;
     
     double fdx = Mu_Cost_f(dmu, p);
     double f   = Mu_Cost_f( mu, p);
    
     
     return (fdx - f) / dstep;
  }

  void Mu_Cost_fdf(double x, void * params, double *f , double *df)
  {

    struct impurity_params *p = static_cast<impurity_params*> (params);

    // struct impurity_params *p = (struct impurity_params *)params;
    double dstep = *(p->dstep);
    double mu = x;
    double dmu = mu + dstep;
    double fdx = Mu_Cost_f(dmu, p);
    
    *f  = Mu_Cost_f(mu, p);
    *df = (fdx - *f)/dstep;


 }

  void print_header_fix_mu_noder( std::ostream& stream )
  {
    auto w   = std::setw(15);
    stream << w << "Nr. Iter" << w 
           << w << "mu-upper" << w
           << w << "mu-lower" << w
           << w << "root"     
           << std::endl;
  }

  void print_state_fix_mu_noder( std::ostream& stream, size_t iter, const gsl_root_fsolver *s )
  {
    double r    = gsl_root_fsolver_root(s);
    double x_lo = gsl_root_fsolver_x_lower(s);
    double x_hi = gsl_root_fsolver_x_upper(s); 

    auto w   = std::setw(15);
    stream << w << iter << w
           << w << x_lo << w
           << w << x_hi << w 
           << w << r    
           << std::endl;
  }

  void print_header_fix_mu_der( std::ostream& stream )
  {
    auto w   = std::setw(15);
    stream << w << "Nr. Iter" << w 
           << w << "mu-current" << w
           << w << "mu-step" << w 
           << std::endl;
  }

  void print_state_fix_mu_der( std::ostream& stream, size_t iter, const gsl_root_fdfsolver *s, double r0 )
  {
    double r    = gsl_root_fdfsolver_root(s);
    auto w   = std::setw(15);
    stream << w << iter << w
           << w << r << w
           << w << r-r0 << w 
           << std::endl;
  }

  const gsl_root_fsolver_type * SelectMuSolver_Type_noder( const std::string &method_name )
  {
      if (method_name == "brent")
          return gsl_root_fsolver_brent;
      else if (method_name == "bisection")
          return gsl_root_fsolver_bisection;
      else if (method_name == "falsepos")
          return gsl_root_fsolver_falsepos;
      else
      {
          std::string msg = "";
          msg += "Error in OptimizeMuEDPot_noder! Passed invalid solver type! Options are: \n";
          msg += " \t(o)     brent: Brent-Dekker bracketing method. \n";
          msg += " \t(o)  falsepos: False position algorithm. \n";
          msg += " \t(o) bisection: Bisection method. \n";
          msg += "Note! All these are algorithms are bracketing methods, not using derivatives!\n";
          throw( std::runtime_error( msg ) );
      }
  }

  const gsl_root_fdfsolver_type * SelectMuSolver_Type_der( const std::string &method_name )
  {
    if (method_name == "newton")
      return gsl_root_fdfsolver_newton;
    else if (method_name == "secant")
      return gsl_root_fdfsolver_secant;
    else if (method_name == "steffenson")
      return gsl_root_fdfsolver_steffenson;
    else
    {
      std::string msg = "";
      msg += "Error in OptimizeMuEDPot_der! Passed invalid solver type! Options are: \n";
      msg += " \t(o)     newton: Newton root-finding method. \n";
      msg += " \t(o)     secant: Secant algorithm. Approximates derivatives. \n";
      msg += " \t(o) steffenson: Steffenson method, the fastest of the three. \n";
      throw( std::runtime_error( msg ) );
    }
  }

  void ProposeInitBracket_MuED( impurity_params *params, double &x_lo, double &x_hi )
  {
    // Try mu = 0, then try to bracket a zero by changing signs.
    double f0 = Mu_Cost_f( 0., params );
    double step = 0.1;
    bool done = false;
    int max_tries = 100, curr_try = 0;
    if( abs(f0) < 1.E-6 )
    {
      x_lo = -0.1;
      x_hi =  0.1;
      done = true;
    }
    else if( f0 > 0. )
    {
    // Too low initial chemical potential
      x_lo = 0.;
      double curr_x = 0.;
      while( curr_try < max_tries )
      {
        curr_try++;
        curr_x = double(curr_try) * step;
        double f = Mu_Cost_f( curr_x, params );
        if( f < 0. )
        {
          done = true;
          break;
        }
      }
      x_hi = curr_x;
    }
    else
    {
    // Too large initial chemical potential
      x_hi = 0.;
      step *= -1;
      double curr_x = 0.;
      while( curr_try < max_tries )
      {
        curr_try++;
        curr_x = double(curr_try) * step;
        double f = Mu_Cost_f( curr_x, params );
        if( f > 0. )
        {
          done = true;
          break;
        }
      }
      x_lo = curr_x;
    }
    // If bracketing failed, try a guess interval
    if( !done )
    {
      x_lo = -10.;
      x_hi =  10.;
    }
  } 

  // Version using derivatives!
  double Fix_Mu_der(const std::string &method_name, double &init_mu, impurity_params * params)
  {


    double abs_tol = *(params -> abs_tol);
    size_t maxiter = *(params->maxiter);
    bool print = *(params->nel_target);

    // double abs_tol;
    // abs_tol =  1.E-4; 
    
    // size_t maxiter;
    // maxiter = 100; 
      
    // bool print;
    // print =  true; 
      
    // Solver type label and solver
      
    const gsl_root_fdfsolver_type *T;
    gsl_root_fdfsolver *s;

    int status; // Status label for optimization steps
    size_t iter = 0; // Iteration counter


    // GSL root function
    gsl_function_fdf f;
    f.f = &Mu_Cost_f;
    f.df = &Mu_Cost_df;
    f.fdf = &Mu_Cost_fdf;
    f.params   = params;



    // Initial bracket for mu, to be 
    double mu0 = init_mu;

    // Set the solver:
    T = SelectMuSolver_Type_der( method_name );
    s = gsl_root_fdfsolver_alloc (T);
    gsl_root_fdfsolver_set (s, &f, mu0);

    // Print header and initial point
    if( print )
    {
      print_header_fix_mu_der( std::cout );
      print_state_fix_mu_der( std::cout, iter, s, 0 );
    }




    // Run optimization
    double mu_prev, mu = mu0;
    do
    {
    	iter++;
      status = gsl_root_fdfsolver_iterate(s);
    	mu_prev = mu;
     	mu      = gsl_root_fdfsolver_root(s);
     	status = gsl_root_test_delta( mu, mu_prev, abs_tol, 1.E-3 );

      if( print )
        print_state_fix_mu_der( std::cout, iter, s, mu_prev );

      if (status == GSL_SUCCESS && print)   // check if solver is stuck
     	  std::cout << "Converged!" << std::endl;
      }
      while (status == GSL_CONTINUE && iter < maxiter);

      // Finally, get and return the optimal chemical potential
      double res_mu = gsl_root_fdfsolver_root( s );
      gsl_root_fdfsolver_free (s);
    
      return res_mu; 
    }


  double Fix_Mu_noder(const std::string &method_name, double &init_mu, impurity_params * params)
  {
    
    // Optimization parameters

    double abs_tol = *(params -> abs_tol);
    size_t maxiter = *(params->maxiter);
    bool print = *(params->nel_target);
    double init_shift = *(params->init_shift);

    // double abs_tol;
    // abs_tol =  1.E-4; 
    // size_t maxiter;
    // maxiter = 100; 
    // bool print;
    // print =  true; 
    // double init_shift ;
    // init_shift =  2.0;

    // Solver type label and solver
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;

    int status; // Status label for optimization steps
    size_t iter = 0; // Iteration counter

    // GSL root function
    gsl_function f;
    f.function = &Mu_Cost_f;
    f.params   = params;

    // Initial bracket for mu, to be 
    double mu0 = init_mu;
    double x_lo = mu0-std::abs(init_shift), x_hi = mu0+std::abs(init_shift);
    ProposeInitBracket_MuED( params, x_lo, x_hi );

    // std::cout<< "f(" << x_lo << ") =" << f.function(x_lo,params) << std::endl;
    // std::cout<< "f(" << x_hi << ") =" << f.function(x_hi,params) << std::endl;


    // Set the solver:
    T = SelectMuSolver_Type_noder( method_name );
    s = gsl_root_fsolver_alloc (T);
    gsl_root_fsolver_set (s, &f, x_lo, x_hi);

    // Print header and initial point
    if( print )
    {
      print_header_fix_mu_noder( std::cout );
      print_state_fix_mu_noder( std::cout, iter, s );
    }


    // Run optimization
    do
    {
      iter++;
      status = gsl_root_fsolver_iterate(s);
      x_lo   = gsl_root_fsolver_x_lower(s);
      x_hi   = gsl_root_fsolver_x_upper(s);
      status = gsl_root_test_interval( x_lo, x_hi, abs_tol, 1.E-4 );


      if( print )
        print_state_fix_mu_noder( std::cout, iter, s );

      if (status == GSL_SUCCESS && print)   // check if solver is stuck
        std::cout << "Converged!" << std::endl;
      }
      while (status == GSL_CONTINUE && iter < maxiter);

      // Finally, get and return the optimal chemical potential
      double res_mu = gsl_root_fsolver_root( s );
      gsl_root_fsolver_free (s);
      return res_mu; 
    }

} // namespace macis