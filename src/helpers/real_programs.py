def get_real_programs(program: str):
    if program == 'if':
        code = """
        a = gm([1],[1.],[2.]);
        if a < 0 {
            b = 3 * a + gm([1],[0.],[1.]);
        } else {
            b = gm([1],[8.],[1.]);  
        } end if;
        """
        return code
    elif program == 'biasedtugwar':
        code = """
        skill1 = gm([1],[20.],[4.]);
        skill2 = gm([1],[20.],[4.]);

        if 1.3*skill2 -skill1 < 0 {
        p1wins = 1;
        } else {
        p1wins = 0;  
        } end if;"""
        return code
    elif program == 'burglary':
        code = """
        earthquake = gm([0.001, 0.999], [1., 0.], [0., 0.]);
        burglary = gm([0.002, 0.998], [1., 0.], [0., 0.]);

        if burglary == 1 {
            if earthquake == 1 {
                alarm = gm([0.95, 0.05], [1, 0], [0, 0]);
            } else {
                alarm = gm([0.94, 0.06], [1, 0], [0, 0]);
            } end if;
        } else {
            if earthquake == 1 {
                alarm = gm([0.29, 0.71], [1, 0], [0, 0]);
            } else {
                alarm = gm([0.001, 0.999], [1, 0], [0, 0]);
            } end if;
        } end if;

        if alarm == 1 {
            johncalls = gm([0.9, 0.1], [1, 0], [0, 0]);
        } else {
            johncalls = gm([0.05, 0.95], [1, 0], [0, 0]);
        } end if;"""
        return code
    elif program == 'csi':
        code = """
        u = gm([0.3, 0.7], [1, 0], [0, 0]);
        v = gm([0.9, 0.1], [1, 0], [0, 0]);
        w = gm([0.1, 0.9], [1, 0], [0, 0]);

        if u == 1 {
            if w == 1{
                x = gm([0.8, 0.2], [1, 0], [0, 0]);
            } else {
                x = gm([0.2, 0.8], [1, 0], [0, 0]);
            } end if;
        } else {
            if v == 1{
                x = gm([0.8, 0.2], [1, 0], [0, 0]);
            } else {
                x = gm([0.2, 0.8], [1, 0], [0, 0]);
            } end if;
        } end if;"""
        return code
    elif program == 'easytugwar':
        code = """
        skill1 = gm([1],[20.],[4.]);
        skill2 = gm([1],[20.],[4.]);

        if skill1 - skill2 > 0 {
            p1wins = 1;
        } else {
        p1wins = 0;  
        } end if;"""
        return code
    elif program == 'eyecolor':
        code = """
        eyecolor = gm([0.8, 0.05, 0.04, 0.01], [0, 1, 2, 3], [0, 0, 0, 0]);

        if eyecolor == 0 {
            haircolor = gm([0.8, 0.05, 0.04, 0.01, 0.1], [0, 1, 2, 3, 4], [0, 0, 0, 0, 0]);
        } else {
            if eyecolor == 1 {
                haircolor = gm([0.7, 0.15, 0.04, 0.01, 0.1], [0, 1, 2, 3, 4], [0, 0, 0, 0, 0]);
            } else {
                if eyecolor == 2 {
                        haircolor = gm([0.4, 0.3, 0.18, 0.02, 0.1], [0, 1, 2, 3, 4], [0, 0, 0, 0, 0]);
                } else {
                        haircolor = gm([0.4, 0.29, 0.18, 0.03, 0.1], [0, 1, 2, 3, 4], [0, 0, 0, 0, 0]);
                } end if;
            } end if;
        } end if;
        hairlenght = gm([0.6, 0.15, 0.25], [0, 1, 2], [0, 0, 0]);"""
        return code
    elif program == 'grass':
        code = """
        rain = gm([1], [4], [2]);

        if rain < 1 {
            sprinkler = 1;
        } else {
            sprinkler = 0;
        } end if;

        if rain > 2 {
            grasswet = 1;
        } else {
            if sprinkler == 1 {
                grasswet = 1;
            } else {
                grasswet = 0;
            }end if;
        }end if;"""
        return code
    elif program == 'hurricane':
        code = """
        preplevel = gm([0.5, 0.2, 0.3], [0, 1, 2], [0, 0, 0]);
        if preplevel == 0 {
            damage = gm([0.2, 0.8], [0, 1], [0, 0]);
        } else {
            if preplevel == 1 {
                damage = gm([0.2, 0.8], [0, 1], [0, 0]);
            } else {
                damage = gm([0.8, 0.2], [0, 1], [0, 0]);
            } end if;
        } end if;"""
        return code
    elif program == 'mixedcondition':
        code = """
        u = bern(0.3);
        v = gm([1],[10.],[2.]);

        if u == 1{
        if v > 12 {
            w = gm([1],[12.],[2.]);
        } else {
            w = gm([1],[6.],[2.]);
        } end if;
        } else {
        w = gm([1],[6.],[2.]);  
        } end if;"""
        return code
    elif program == 'mog1':
        code = """
        mu = gm([1], [20.], [3.]);
        sigma = gm([1], [2.], [1.]);
        x = sigma * gm([1], [0.], [1.]);
        x = x + mu;"""
        return code
    elif program == 'multiplebranches':
        code = """
        contentDifficulty = gm([1], [30.], [5.]);
        if contentDifficulty < 35{
        if contentDifficulty < 20 {
            questionsAfterLectureLength = gm([1], [2.], [1.]);
        } else {
            questionsAfterLectureLength = gm([1], [10.], [3.]);
        } end if;
        } else {
        questionsAfterLectureLength = gm([1], [25.], [6.]); 
        } end if;"""
        return code
    else:
        raise ValueError(f"Unknown program type: {program}")
