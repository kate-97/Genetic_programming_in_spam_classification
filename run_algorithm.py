from genetic_programming import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, \
    recall_score, precision_score, confusion_matrix
from matplotlib import pyplot as plt

# TODO: Evaluation and visualisation of working of the algorithm
# TODO: Documentation, presentation
# TODO: Parallel testing of the scripts.
# TODO: Comparison with other classification algorithms.


def feature_selection(data):
    features = data.columns
    unused_features = [attr for attr in list(data.columns) if
                       data[data['class'] == 0][attr].mean() > data[data['class'] == 1][attr].mean()]

    used_features = [feature for feature in features if feature not in unused_features]
    used_features.remove('class')

    return used_features


def execute_genetic_algorithm(data, population_size, pm, elitism_number,
                              number_of_iterations, output=True, logf=None):
    print("#" * 20 + "\n")
    print("   Number of iterations: " + str(number_of_iterations))
    print("   Population size: " + str(population_size))
    print("   Probability of mutation (pm) : " + str(pm))
    print("   Number of elite individuals: " + str(elitism_number))

    if logf is not None:
        logf.write("#" * 20 + "\n")
        logf.write("\t Number of iterations: " + str(number_of_iterations) + "\n")
        logf.write("\t Population size: " + str(population_size) + "\n")
        logf.write("\t Probability of mutation (pm) : " + str(pm) + "\n")
        logf.write("\t Number of elite individuals: " + str(elitism_number) + "\n")

    used_features = feature_selection(data)

    population = initialize_population(data, used_features, population_size)
    next_population = copy.deepcopy(population)

    current_iteration = 0

    while current_iteration < number_of_iterations:
        if current_iteration == number_of_iterations // 4 or \
          current_iteration == number_of_iterations // 2 or \
          current_iteration == 3 * number_of_iterations // 4:
            print("-- " + str((current_iteration / number_of_iterations) * 100)
                  + "% done")

        for individual in population:
            individual.calculate_fitness(data)

        # print("Found fit")

        population.sort()

        i = 0
        for i in range(0, elitism_number):
            next_population[i] = population[i]

        i = elitism_number
        while i < population_size - 1:
            crossing = np.random.choice([True, False], p=[1 - pm, pm])

            if not crossing:
                selected_individual = selection(population)  # selekcija jedne jedinke (fitnes-srazmerna iliti ruletska)
                next_population[i] = mutate(selected_individual)
                i += 1

            #    ako ipak ukrstanje
            else:
                selected_individual1 = selection(population)
                selected_individual2 = selection(population)
                #      selekcija dve jedinke (rulet)
                #      cross over

                cross_over(selected_individual1, selected_individual2, next_population[i], next_population[i + 1])

        population = next_population
        current_iteration += 1

    average_fit = sum([individual.fitness for individual in population]) / len(population)

    if output:
        print("Algorithm finished. \n " +
              "The best individual has value of fitness: "
              + str(population[0].fitness) + "\n" +
              "Average value of fitness value is: " +
              str(average_fit))

    if logf is not None:
        logf.write("\n\n ** The best value of fitness function in population:\n" +
                   " (that is, the best accuracy in population): " + str(population[0].fitness) + "\n"
                   )
        logf.write("Average fitness (average accuracy) is: " + str(average_fit) + "\n")
        logf.write("#" * 20 + "\n")
        logf.flush()

    return population


# TODO: Eventualno jos neku vizualizaciju
def evaluate_classifier(classifier, data, logf=None):
    all_instances = data.shape[0]

    predicted_classes = []
    true_classes = []

    for i in range(all_instances):
        predicted = classifier.classify_mail(data.iloc[i, :])
        true_class = data.iloc[i, :]['class']

        predicted_classes.append(predicted)
        true_classes.append(true_class)

    y_true = pd.Series(true_classes)
    y_predicted = pd.Series(predicted_classes)

    print("***** Classification results: *****")
    accuracy = accuracy_score(y_true, y_predicted)
    print("   Accuracy: " + str(accuracy))

    precision = precision_score(y_true, y_predicted)
    print("   Precision: " + str(precision))

    f1 = f1_score(y_true, y_predicted)
    print("   F score: " + str(f1))

    matrix_conf = confusion_matrix(y_true, y_predicted)
    print("Matrix confusion: ")
    print(str(matrix_conf))
    print("**********")

    if logf is not None:
        logf.write("***** Classification results: *****\n")
        logf.write("   Accuracy: " + str(accuracy) + "\n")
        logf.write("   Precision: " + str(precision) + "\n")
        logf.write("   F score: " + str(f1) + "\n")
        logf.write("Matrix confusion: \n\n")
        logf.write(str(matrix_conf))
        logf.write("**********")

    return y_predicted


def examine_genetic_algorithm(n,
                              data, used_features, population_size, pm, elitism_number,
                              number_of_iterations, output=False, logf = None):

    the_best_fitness = 0.0
    iteration_the_best_fitness = 0
    average_the_best_fitness = 0.0

    the_best_average_fitness = 0.0
    iteration_the_best_average_fitness = 0
    best_fitness_the_best_average = 0.0

    for i in range(n):
        population = execute_genetic_algorithm(data, used_features, population_size, pm, elitism_number,
                                               number_of_iterations, output, logf)
        if population[0].fitness > the_best_fitness:
            the_best_fitness = population[0].fitness
            iteration_the_best_fitness = i
            average_the_best_fitness = sum([individual.fitness for individual in population]) / len(population)

        current_average = sum([individual.fitness for individual in population]) / len(population)
        if current_average > the_best_average_fitness:
            the_best_average_fitness = current_average
            iteration_the_best_average_fitness = i
            best_fitness_the_best_average = population[0].fitness

    print("After " + str(n) + "executions: \n")
    print("\t\t The best value of fitness: " + str(the_best_fitness))
    print("\t\t Iteration in which was this value of fitness: " + str(iteration_the_best_fitness))
    print("\t\t Average value of fitness in this iteration: " + str(average_the_best_fitness) +"\n")
    print("\t\t The best average value of fitness: " + str(the_best_average_fitness))
    print("\t\t Iteration which was this average value of fitness: " + str(iteration_the_best_average_fitness))
    print("\t\t The best value of fitness in this iteration: " + str(best_fitness_the_best_average) + "\n")

    return the_best_fitness


def visualise_results(i, y_true, y_predicted):
    matrix_conf = confusion_matrix(y_true, y_predicted)
    plt.imshow(matrix_conf)
    plt.colorbar()
    plt.xticks(range(2), ['Ham', 'Spam'])
    plt.yticks(range(2), ['Ham', 'Spam'])
    plt.savefig('figure_' + str(i) + '.png', format='png')
    plt.show()


def ensemble_classification(population, instance):
    for_positive = 0
    for_negative = 0

    for individual in population:
        assigned_class = individual.chromosome.classify_mail(instance)

        if assigned_class == 1:
            for_positive += 1
        else:
            for_negative += 1

    if for_positive > for_negative:
        return 1
    else:
        return 0


def evaluate_ensemble(population, data):
    print ("** Ensemble results **")
    all_instances = data.shape[0]
    predicted_classes = []

    for i in range(all_instances):
        predicted_classes.append(ensemble_classification(population,
            data.iloc[i,:]))

    print("Accuracy: " + str(accuracy_score(data['class'], predicted_classes)))
    print("Precision: " + str(precision_score(data['class'], predicted_classes)))
    print("F score: " + str(f1_score(data['class'], predicted_classes)))
    print("Matrix confusion: \n" + str(confusion_matrix(data['class'], predicted_classes)))
    print("****")


def execute_and_evaluate_the_best(data, population_size, pm, elitism_number,
                                  number_of_iterations, i, logf=None):
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)

    population = execute_genetic_algorithm(X_train, population_size, pm, elitism_number,
                                           number_of_iterations, output=False, logf=logf)

    y_test_predicted = evaluate_classifier(population[0].chromosome, X_test, logf)
    visualise_results(i, y_test, y_test_predicted)
    evaluate_ensemble(population[:(elitism_number//4)], pd.DataFrame(X_test, columns=data.columns))


def demonstrate_without_gp(n, data, used_features, logf=None):
    sum_of_fitnesses = 0.0
    max_of_fitnesses = 0.0

    all_instances = data.shape[0]

    for j in range(n):
        correct = 0
        s = decision_tree(used_features)
        for i in range(all_instances):
            predicted = s.classify_mail(data.iloc[i, :])
            true_class = int(data.iloc[i, :]['class'])

            if predicted == true_class:
                correct += 1
        sum_of_fitnesses += correct / all_instances

        if max_of_fitnesses < correct / all_instances:
            max_of_fitnesses = correct / all_instances

    print("Average accuracy: " + str(sum_of_fitnesses / n))
    print("The best: " + str(max_of_fitnesses))

    if logf is not None:
        logf.write("~"*20 + "\n")
        logf.write("\t Execution " + str(n) + " times without GP: \n")
        logf.write("Average accuracy: " + str(sum_of_fitnesses / n) + "\n")
        logf.write("The best: " + str(max_of_fitnesses) + "\n")
        logf.write("~" * 20 + "\n")

