# ensamble.jl

Automatic Test Assembly (ATA)  in julia and a MIP solver

## The Software

This repository presents an example of how the process of selecting items for a test may be implemented using free software. For now, it is only an early and incomplete example and is not recommended for use in high-stakes applications.

## Overview

In the psychometric tradition of standardized testing, a single version of a test is often deemed sufficient. This is "the" test, the result of months or even years of research, trial, and error. Once this test is approved, it is administered to a reference population and a standard is established, which further serves to classify individuals relative to this original population, the "normative" group. However, there are numerous reasons why this approach may not be practical in educational contexts. Consider the implications if the same version of the SAT were used repeatedly—year after year, or for every new cohort of students.

Assembling two or more forms of a test is exponentially more complex than creating just one. Not only must the content specifications be meticulously adhered to —already a hard task— but the difficulty or information levels of the items must also be comparable. Otherwise, one form may be easier or noisier than another, with potentially severe consequences for test-takers. It would be highly inequitable for a student's score to depend on which version they received, as it is simply unfair for one to fail entry into a university because they took the harder version.

Item Response Theory (IRT) offers a common solution to this problem, Once the statistical characteristics of the items are known from prior testing, their future performance can be predicted with some high accuracy. This allows for the assembly of a new test version with a reasonably clear idea of how it will perform operationally. Classical Test Theory can also be used, but it is more challenging to obtain difficulty estimates on the same scale, leading to potentially unexpected results.

Traditionally, this process is iterative —manually combining available items from the bank like puzzle pieces according to their content and estimated difficulties, until the objective is met. Manual assembly is tedious; it can take hours, or even days in some cases, especially when assembling multiple booklets with dozens of items each. This involves a process of adding and removing items until the desired outcome is reached.

Computers have brought significant advantages to this process. To ease this burden and minimize the errors that result from manual processes, optimization methods have increasingly been used to enable the relatively automatic assembly of test versions.

The idea is that if the test constructor defines the desired characteristics of the test in detail, including the necessary topics and content, the machine can select the best combinations.

(A key reference for this methodology is the book Linear Models for Optimal Test Design by Wim Van der Linden (2005). This book discusses the models covered here, as well as others applicable to different scenarios, from assembling a single test version to designing block-sampling for evaluating entire populations, and even constructing item banks for an assessment program and adaptive test design.)

## Advantages of Automatic Assembly

- Speed and Efficiency: Test assembly becomes a much faster and less tedious process.

- Objective and Reproducible: It forces the creation of well-defined and detailed specifications, making the process more objective and reproducible.

- Simultaneous Version Assembly: Multiple versions can be assembled simultaneously. This also promotes better use of items.

- Powerful Constraints Management: The process is robust enough to meet all constraints objectively, even with complex specification tables.

- Customizable: It is possible to assemble specific versions according to the needs of each application. In some cases, it is even possible to create and modify the test version "on the fly" for each test-taker, adapting the content as they respond.

- Automatic Reporting: The system can generate detailed and fast assembly reports automatically.

## Disadvantages

- Rigorous Specifications Required: Although specification tables and assembly rules should always exist, they are sometimes incomplete or allow too much discretion to the test constructor. While this flexibility is appreciated by many who view test assembly as an art, automated assembly eliminates this margin. Good or bad as this may be, significant effort is required in ATA to ensure that all rules are well-defined, detailed, and codified so that the machine can process them effectively. Even tolerances must be explicit.

- Possible Non-Solutions: The program may not always find a satisfactory solution, especially when assembly conditions are highly complex or the rules are contradictory, which happens often. In such cases, additional effort is needed to identify and correct the problems after each failed attempt.

- Uncertain Outcomes: Like manual processes, the final versions do not always perform as expected in operational use. Whether manual or automatic, assembly is not a magical, infallible solution, and quality assurance methods and post-application checks, as well as routine scaling or equating procedures, are always necessary.

## Calibrating the Item Bank

This step is not strictly part of the assembly process, but it is a fundamental prerequisite; you cannot proceed without a well-calibrated item bank with sufficient sample data. Various specialized software packages can be used for item calibration. Some, like Winsteps, Facets, or ConQuest for the Rasch model family, have a long history in the market. For Item Response Theory, other options include Bilog, Parscale, Multilog, or FlexMIRT. There is also a growing number of packages for exploring new models, primarily developed for the R programming language, such as TAM, SIRT, MIRT, ErM, and many others available on CRAN.

## Optimization

There is a number of general optimization software that can be easily adapted for our needs. The most common technique is "Mixed-Integer Programming" (MIP), which is used here. This software tends to find the best combinations, although in complex cases it may take too long or fail to reach a solution. For this reason, other methodologies are sometimes used, such as simulated annealing, genetic algorithms, constraint programming, network-flow algorithms, and even  Markov chains. Linear optimization programs tend to be more precise and efficient, especially when additional constraints are involved (such as including various topics, item types, or considering the presence of friend and enemy items).

## Julia and JuMP

Julia is a high-level language that compiles to very efficient code and can be used interactively, which facilitates work. It can be compared to Python in simplicty and easy of use and, in many situations, it has a performance similar to the fastes compiled programs. It is well suited for use in numerical and data analysis problems.

JuMP, on the other hand, is a Julia-based package that provides tools for formulating optimization models to be used with a long list of solvers (commercial and open source). In this case, the models were tested with almost identical results using:

- IBM CPLEX
- Cbc (coin-or)
- SCIP
- GLPK
- HiGHS

## Alternatives

There are other ways to achieve this. In R, for example, the package TestDesign seems to be a good solution that saves several steps and requires little programming. Other examples include eatATA, ATA, xxIRT, dexterMST, catR, mstR —all of them in R, perhaps the most popular language for this purpose. Some of these packages are designed for assembling adaptive tests.

In Julia, Python, and SAS, there are interesting, though somewhat unpolished, solutions that require at least basic knowledge of the underlying programming languages. In a way, these can be considered experimental libraries. Major testing and assessment agencies typically develop their own in-house solutions.
