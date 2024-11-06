for submodule in $(git ls-files --stage | grep 160000 | cut -f 2); do
    echo "Limpiando submódulo roto: $submodule"
    git rm --cached $submodule || echo "No se pudo limpiar $submodule"
    rm -rf .git/modules/$submodule
done

